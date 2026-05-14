"""Local speech-to-text frontend for realtime response-audio backends."""

import os
import sys
import time
import asyncio
import logging
from typing import Any, Tuple
from pathlib import Path
from concurrent.futures import Future as ConcurrentFuture

import numpy as np
from fastrtc import AdditionalOutputs, audio_to_float32
from numpy.typing import NDArray
from scipy.signal import resample
from opentelemetry import trace
from opentelemetry import context as otel_context
from openai.types.realtime import (
    RealtimeAudioConfigParam,
    RealtimeAudioConfigOutputParam,
    RealtimeResponseCreateParamsParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM

from robot_comic import telemetry
from robot_comic.pause import TranscriptDisposition
from robot_comic.config import (
    HF_BACKEND,
    OPENAI_BACKEND,
    config,
    get_default_voice_for_backend,
)
from robot_comic.prompts import get_session_voice, get_session_instructions
from robot_comic.welcome_gate import GateState, WelcomeGate
from robot_comic.base_realtime import to_realtime_tools_config
from robot_comic.openai_realtime import OpenaiRealtimeHandler
from robot_comic.tools.core_tools import ToolDependencies, get_active_tool_specs
from robot_comic.huggingface_realtime import HuggingFaceRealtimeHandler


logger = logging.getLogger(__name__)

LOCAL_STT_SAMPLE_RATE = 16000


class LocalSTTDependencyError(RuntimeError):
    """Raised when the optional local STT dependency is not installed."""


def resolve_ort_model_path(model_path: Path) -> tuple[Path, str]:
    """Return (resolved_path, format_str) preferring .ort over .onnx.

    If *model_path* is a directory, we look for any ``.ort`` file inside it
    first; if none is found we look for any ``.onnx`` file and return that
    (falling back to the original path so the caller can pass it straight to
    ``Transcriber``).  If *model_path* is a file, we just swap the suffix.

    The returned *format_str* is ``"ort"`` or ``"onnx"`` and is used only for
    the startup log line.
    """
    if model_path.is_dir():
        ort_files = sorted(model_path.glob("*.ort"))
        if ort_files:
            return ort_files[0], "ort"
        onnx_files = sorted(model_path.glob("*.onnx"))
        if onnx_files:
            return model_path, "onnx"
        return model_path, "onnx"

    ort_candidate = model_path.with_suffix(".ort")
    if ort_candidate.exists():
        return ort_candidate, "ort"
    return model_path, "onnx"


def prewarm_model_file(model_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Pull the model file into the OS page cache by reading it sequentially.

    This is a no-op on Windows where the page-cache concept does not apply in
    the same way and the overhead is not worthwhile.
    """
    if sys.platform == "win32":
        return
    resolved = model_path if model_path.is_file() else model_path
    # If model_path is a directory, prewarm the first .ort or .onnx file found.
    if resolved.is_dir():
        candidates = sorted(resolved.glob("*.ort")) or sorted(resolved.glob("*.onnx"))
        if not candidates:
            return
        resolved = candidates[0]
    if not resolved.is_file():
        return
    try:
        fd = os.open(str(resolved), os.O_RDONLY)
        try:
            while True:
                chunk = os.read(fd, chunk_size)
                if not chunk:
                    break
        finally:
            os.close(fd)
        logger.debug("Moonshine page-cache prewarm complete: %s", resolved)
    except OSError as exc:
        logger.debug("Moonshine prewarm skipped (%s): %s", resolved, exc)


def _resolve_moonshine_arch(moonshine_voice: Any, model_name: str) -> Any:
    """Return a Moonshine ModelArch value, falling back to the package default."""
    model_arch = getattr(moonshine_voice, "ModelArch", None)
    if model_arch is None:
        return None
    arch_name = model_name.upper()
    if not arch_name.endswith("_STREAMING"):
        arch_name = f"{arch_name}_STREAMING"
    return getattr(model_arch, arch_name, None)


class _MoonshineListener:  # base class is attached dynamically in _build_local_stt_stream()
    """Bridge Moonshine transcript callbacks into the asyncio handler loop."""

    def __init__(self, handler: "LocalSTTInputMixin") -> None:
        self.handler = handler

    def on_line_started(self, event: Any) -> None:
        text = self._text_from_event(event)
        self.handler._heartbeat.update(
            {"state": "speech_started", "last_event": "started", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("started", text)

    def on_line_updated(self, event: Any) -> None:
        text = self._text_from_event(event)
        self.handler._heartbeat.update(
            {"state": "partial", "last_event": "partial", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("partial", text)

    def on_line_text_changed(self, event: Any) -> None:
        text = self._text_from_event(event)
        self.handler._heartbeat.update(
            {"state": "partial", "last_event": "partial", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("partial", text)

    def on_line_completed(self, event: Any) -> None:
        text = self._text_from_event(event)
        self.handler._heartbeat.update(
            {"state": "completed", "last_event": "completed", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("completed", text)

    def on_error(self, event: Any) -> None:
        logger.warning("Local STT error: %s", getattr(event, "error", event))

    @staticmethod
    def _text_from_event(event: Any) -> str:
        line = getattr(event, "line", event)
        text = getattr(line, "text", "")
        return text if isinstance(text, str) else str(text)


class LocalSTTInputMixin:
    """Mixin that replaces upstream mic-audio transport with local Moonshine STT."""

    # Declared here so mypy sees them on the mixin; the concrete base class
    # provides the actual values via super().__init__().
    deps: ToolDependencies
    output_queue: asyncio.Queue[Any]

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize local STT state and the response backend."""
        super().__init__(deps, sim_mode, instance_path, startup_voice)  # type: ignore[call-arg]
        # Declare timing / span attrs so mypy sees consistent types on the mixin.
        # Concrete base classes (BaseLlamaResponseHandler, BaseRealtimeHandler)
        # initialise these in their own __init__ via super().__init__() above.
        self._turn_user_done_at: float | None
        self._turn_response_created_at: float | None
        self._turn_first_audio_at: float | None
        self._turn_id: str | None
        self._turn_start_at: float | None
        self.local_stt_sample_rate = LOCAL_STT_SAMPLE_RATE
        self._local_stt_stream: Any = None
        self._local_stt_transcriber: Any = None
        self._local_stt_listener: Any = None
        self._local_loop: asyncio.AbstractEventLoop | None = None
        self._last_completed_transcript: str = ""
        self._last_completed_at: float = 0.0
        self._heartbeat: dict[str, Any] = {
            "state": "idle",
            "last_event": None,
            "last_text": "",
            "last_event_at": time.monotonic(),
            "audio_frames": 0,
        }
        self._heartbeat_future: "asyncio.Future[Any] | ConcurrentFuture[None] | None" = None
        self._welcome_gate: WelcomeGate | None = self._build_welcome_gate()

    def _build_welcome_gate(self) -> WelcomeGate | None:
        """Create a WelcomeGate for the current profile when the feature is enabled.

        Returns None when REACHY_MINI_WELCOME_GATE_ENABLED is false or no
        profile directory can be resolved.
        """
        if not getattr(config, "WELCOME_GATE_ENABLED", False):
            return None

        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        profiles_dir = getattr(config, "PROFILES_DIRECTORY", None)
        if not profile or profiles_dir is None:
            logger.info("welcome gate: enabled but no profile selected — gate inactive")
            return None

        from robot_comic.welcome_gate import make_gate_for_profile

        profile_dir = Path(profiles_dir) / profile
        gate = make_gate_for_profile(profile_dir)
        logger.info("welcome gate: active for profile %r", profile)
        return gate

    async def _prepare_startup_credentials(self) -> None:
        """Let the response backend prepare itself, then initialize local STT."""
        await super()._prepare_startup_credentials()  # type: ignore[misc]
        self._local_loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._build_local_stt_stream)

    def _build_local_stt_stream(self) -> None:
        """Build and start the Moonshine streaming transcriber."""
        try:
            import moonshine_voice
            from moonshine_voice import Transcriber, TranscriptEventListener, get_model_for_language
        except Exception as e:  # pragma: no cover - exercised via unit tests without dependency
            raise LocalSTTDependencyError(
                "Local STT requires the optional dependency group: install with `uv sync --extra local_stt` "
                "or `pip install -e .[local_stt]`."
            ) from e

        language = (getattr(config, "LOCAL_STT_LANGUAGE", None) or "en").strip().lower()
        model_name = getattr(config, "LOCAL_STT_MODEL", "tiny_streaming")
        update_interval = float(getattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35))
        requested_arch = _resolve_moonshine_arch(moonshine_voice, model_name)
        cache_root = Path(getattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")).expanduser()

        try:
            model_path, model_arch = get_model_for_language(
                language,
                wanted_model_arch=requested_arch,
                cache_root=cache_root,
            )
        except TypeError:
            model_path, model_arch = get_model_for_language(language, cache_root=cache_root)
            model_arch = requested_arch or model_arch

        # Prefer .ort (ONNX Runtime FlatBuffer) when available — loads ~3-5x
        # faster than plain .onnx because it skips the graph parse step.
        model_path, model_format = resolve_ort_model_path(Path(model_path))

        # Pull the model file into the OS page cache before handing it to
        # ONNX Runtime.  No-op on Windows.
        prewarm_model_file(model_path)

        logger.info(
            "Starting local Moonshine STT: language=%s model=%s format=%s update_interval=%.2fs cache=%s path=%s",
            language,
            model_name,
            model_format,
            update_interval,
            cache_root,
            model_path,
        )

        # moonshine_voice.Transcriber expects the *directory* containing the
        # model file and its tokenizer.bin sibling, not the model file itself.
        transcriber_path = model_path.parent if model_path.is_file() else model_path
        transcriber = Transcriber(
            model_path=str(transcriber_path), model_arch=model_arch, update_interval=update_interval
        )
        stream = transcriber.create_stream(update_interval=update_interval)
        listener_cls = type(
            "RobotComicMoonshineListener",
            (_MoonshineListener, TranscriptEventListener),
            {},
        )
        listener = listener_cls(self)
        stream.add_listener(listener)
        stream.start()

        self._local_stt_transcriber = transcriber
        self._local_stt_stream = stream
        self._local_stt_listener = listener

        if config.MOONSHINE_HEARTBEAT and self._local_loop is not None:
            self._heartbeat_future = asyncio.run_coroutine_threadsafe(
                self._moonshine_heartbeat_loop(), self._local_loop
            )

    def _schedule_local_stt_event(self, kind: str, text: str) -> None:
        """Schedule a local STT event onto the handler event loop."""
        loop = self._local_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(self._handle_local_stt_event(kind, text)))

    async def _handle_local_stt_event(self, kind: str, text: str) -> None:
        """Handle local STT lifecycle events inside the asyncio loop."""
        import uuid as _uuid

        transcript = (text or "").strip()
        if kind == "started":
            self._mark_activity("local_stt_speech_started")  # type: ignore[attr-defined]
            self._turn_user_done_at = None
            self._turn_response_created_at = None
            self._turn_first_audio_at = None
            # Open root turn span and STT infer child span
            if hasattr(self, "_close_turn_span"):
                self._close_turn_span("interrupted")
            _now = time.perf_counter()
            _tracer = telemetry.get_tracer()
            _turn_id = str(_uuid.uuid4())
            if hasattr(self, "_turn_id"):
                self._turn_id = _turn_id
            if hasattr(self, "_turn_start_at"):
                self._turn_start_at = _now
            if hasattr(self, "_session_id"):
                _session_id = self._session_id
            else:
                _session_id = ""
            if hasattr(self, "_current_response_has_tool_call"):
                self._current_response_has_tool_call = False
            _turn_span = _tracer.start_span(
                "turn",
                attributes={
                    "turn.id": _turn_id,
                    "session.id": _session_id,
                    "robot.mode": "local_stt",
                },
            )
            if hasattr(self, "_turn_span"):
                self._turn_span = _turn_span
            _ctx_token = otel_context.attach(trace.set_span_in_context(_turn_span))
            if hasattr(self, "_turn_ctx_token"):
                self._turn_ctx_token = _ctx_token
            _prior_stt = getattr(self, "_stt_infer_span", None)
            if _prior_stt is not None:
                _prior_stt.end()
            self._stt_infer_span: Any = _tracer.start_span("stt.infer")
            self._stt_infer_start: float = _now
            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            return

        if kind == "partial":
            if transcript:
                self._mark_activity("local_stt_partial")  # type: ignore[attr-defined]
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
            return

        if kind != "completed" or not transcript:
            return

        now = time.perf_counter()
        if transcript == self._last_completed_transcript and now - self._last_completed_at < 0.75:
            logger.debug("Ignoring duplicate local STT completion: %s", transcript)
            return
        self._last_completed_transcript = transcript
        self._last_completed_at = now

        self._mark_activity("local_stt_completed")  # type: ignore[attr-defined]
        self.deps.movement_manager.set_listening(False)
        words = transcript.split()
        excerpt = " ".join(words[:5]) + ("…" if len(words) > 5 else "")
        stt_span = getattr(self, "_stt_infer_span", None)
        if stt_span is not None:
            stt_s = now - getattr(self, "_stt_infer_start", now)
            # Tag the stt.infer span with the excerpt *before* closing it so the
            # monitor can read it from the completed child span while the outer
            # turn span is still open (pending row).
            stt_span.set_attribute("turn.excerpt", excerpt)
            stt_span.end()
            self._stt_infer_span = None
            telemetry.record_stt(stt_s, {"gen_ai.system": "local_stt", "stt.type": "moonshine"})
        # Tag the outer turn span with the same excerpt for the completed-row path.
        _outer_span = getattr(self, "_turn_span", None)
        if _outer_span is not None:
            _outer_span.set_attribute("turn.excerpt", excerpt)
        self._turn_user_done_at = now
        self._turn_response_created_at = None
        self._turn_first_audio_at = None

        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        pause_controller = getattr(self.deps, "pause_controller", None)
        if pause_controller is not None:
            try:
                disposition = pause_controller.handle_transcript(transcript)
            except Exception as e:
                logger.error("pause_controller.handle_transcript raised: %s", e)
                disposition = TranscriptDisposition.DISPATCH
            if disposition is TranscriptDisposition.HANDLED:
                return

        # Echo guard: suppress transcripts that arrive while TTS is still playing.
        # Pause commands are intentionally exempted (handled above).
        speaking_until = getattr(self, "_speaking_until", 0.0)
        if time.perf_counter() < speaking_until:
            logger.info("Echo guard: discarding transcript during TTS playback: %r", transcript[:60])
            return

        # Welcome-gate: if enabled and still WAITING, only check for the wake-name.
        # Do NOT dispatch to the LLM until the gate opens.
        gate = self._welcome_gate
        if gate is not None and gate.state is GateState.WAITING:
            if not gate.consider(transcript):
                logger.debug("welcome gate: WAITING — transcript not dispatched: %r", transcript[:60])
                return
            # Gate just opened — fall through and dispatch this transcript normally.

        await self._dispatch_completed_transcript(transcript)

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """Send a completed transcript to the realtime response backend.

        Override in subclasses to redirect to a different response backend.
        """
        if not self.connection:  # type: ignore[attr-defined]
            logger.debug("Local STT transcript ready but realtime connection is not connected")
            return

        await self.connection.conversation.item.create(  # type: ignore[attr-defined]
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": transcript}],
            },
        )
        await self._safe_response_create(  # type: ignore[attr-defined]
            response=RealtimeResponseCreateParamsParam(
                instructions="Answer the user's transcribed speech naturally and concisely in audio.",
            ),
        )

    def _log_heartbeat(self) -> None:
        """Emit one heartbeat log line with current Moonshine state."""
        h = self._heartbeat
        age = time.monotonic() - h["last_event_at"]
        text_snippet = (h["last_text"] or "")[:40]
        logger.info(
            "[Moonshine] state=%s  last_event=%s  age=%.1fs  frames=%d  text=%r",
            h["state"],
            h["last_event"],
            age,
            h["audio_frames"],
            text_snippet,
        )
        if h["state"] == "idle" and age > 10.0 and h["audio_frames"] > 0:
            logger.warning(
                "[Moonshine] idle for %.1fs with %d audio frames received — possible thread-lock or model stall",
                age,
                h["audio_frames"],
            )

    async def _moonshine_heartbeat_loop(self) -> None:
        """Log Moonshine state every second while the stream is active."""
        while self._local_stt_stream is not None and config.MOONSHINE_HEARTBEAT:
            self._log_heartbeat()
            await asyncio.sleep(1.0)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Feed microphone audio into the local STT stream."""
        if self._local_stt_stream is None:
            return

        # Skip pumping mic audio while the robot is speaking. Without this,
        # Moonshine's streaming VAD treats the robot's own TTS as one
        # continuous utterance and never emits `completed`, stalling the
        # next user turn. The post-transcript echo guard alone is too late.
        speaking_until = getattr(self, "_speaking_until", 0.0)
        if time.perf_counter() < speaking_until:
            return

        input_sample_rate, audio_frame = frame
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        audio_float = audio_to_float32(audio_frame)

        if self.local_stt_sample_rate != input_sample_rate:
            audio_float = resample(
                audio_float,
                int(len(audio_frame) * self.local_stt_sample_rate / input_sample_rate),
            ).astype(np.float32, copy=False)
        try:
            self._local_stt_stream.add_audio(audio_float.tolist(), self.local_stt_sample_rate)
            self._heartbeat["audio_frames"] += 1
        except Exception as e:
            logger.debug("Dropping local STT audio frame: %s", e)

    async def shutdown(self) -> None:
        """Shutdown realtime and local STT resources."""
        if self._heartbeat_future is not None:
            self._heartbeat_future.cancel()
            self._heartbeat_future = None
        await super().shutdown()  # type: ignore[misc]
        stream = self._local_stt_stream
        transcriber = self._local_stt_transcriber
        self._local_stt_stream = None
        self._local_stt_transcriber = None
        self._local_stt_listener = None

        def _close_local() -> None:
            for obj, methods in (
                (stream, ("stop", "close")),
                (transcriber, ("close",)),
            ):
                if obj is None:
                    continue
                for method_name in methods:
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as e:
                            logger.debug("Local STT %s failed during shutdown: %s", method_name, e)

        await asyncio.to_thread(_close_local)


class LocalSTTOpenAIRealtimeHandler(LocalSTTInputMixin, OpenaiRealtimeHandler):
    """Use local Moonshine STT for input and OpenAI Realtime for responses."""

    BACKEND_PROVIDER = OPENAI_BACKEND
    SAMPLE_RATE = 24000
    REFRESH_CLIENT_ON_RECONNECT = False
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = OpenaiRealtimeHandler.AUDIO_OUTPUT_COST_PER_1M
    TEXT_INPUT_COST_PER_1M = OpenaiRealtimeHandler.TEXT_INPUT_COST_PER_1M
    TEXT_OUTPUT_COST_PER_1M = OpenaiRealtimeHandler.TEXT_OUTPUT_COST_PER_1M
    IMAGE_INPUT_COST_PER_1M = OpenaiRealtimeHandler.IMAGE_INPUT_COST_PER_1M

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured OpenAI voice for the local-STT response backend."""
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Return a text-in/audio-out realtime config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=get_session_instructions(),
            audio=RealtimeAudioConfigParam(
                output=RealtimeAudioConfigOutputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    def get_current_voice(self) -> str:
        """Return the OpenAI voice selected for local-STT responses."""
        default_voice = get_default_voice_for_backend(OPENAI_BACKEND)
        return self._voice_override or self._get_session_voice(default=default_voice)


class LocalSTTHuggingFaceRealtimeHandler(LocalSTTInputMixin, HuggingFaceRealtimeHandler):
    """Use local Moonshine STT for input and Hugging Face realtime for responses."""

    BACKEND_PROVIDER = HF_BACKEND
    SAMPLE_RATE = 16000
    REFRESH_CLIENT_ON_RECONNECT = True
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = 0.0
    TEXT_INPUT_COST_PER_1M = 0.0
    TEXT_OUTPUT_COST_PER_1M = 0.0
    IMAGE_INPUT_COST_PER_1M = 0.0


# Backward-compatible name for tests/imports; OpenAI remains the default response backend.
LocalSTTRealtimeHandler = LocalSTTOpenAIRealtimeHandler
