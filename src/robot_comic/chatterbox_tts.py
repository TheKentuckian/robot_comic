"""Chatterbox TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls an
Ollama LLM via /api/chat for text generation and tool dispatch, then synthesises
audio with the Chatterbox TTS server's /tts endpoint (voice cloning mode).

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
"""

import re
import json
import uuid
import asyncio
import logging
from typing import Any, Optional

import httpx
import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item
from scipy.signal import resample

from robot_comic.config import (
    CHATTERBOX_OUTPUT,
    CHATTERBOX_DEFAULT_URL,
    CHATTERBOX_DEFAULT_VOICE,
    CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_TEMPERATURE,
    CHATTERBOX_DEFAULT_EXAGGERATION,
    CHATTERBOX_DEFAULT_GAIN,
    config,
    set_custom_profile,
)
from robot_comic.prompts import get_session_instructions
from robot_comic.tools.core_tools import ToolDependencies, get_active_tool_specs
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.chatterbox_tag_translator import translate
from robot_comic.tools.background_tool_manager import ToolCallRoutine, ToolNotification, BackgroundToolManager


logger = logging.getLogger(__name__)

_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400          # 100 ms at 24 kHz
_CHATTERBOX_SAMPLE_RATE = 24000
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0
_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5

# Split at whitespace that follows a sentence-ending punctuation mark.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Hermes3 sometimes emits tool calls as plain text: {function:name, key:val, ...}
_TEXT_TOOL_CALL_RE = re.compile(
    r"^\s*\{\s*function\s*:\s*(\w+)\s*(?:,\s*(.*?))?\s*\}\s*$",
    re.DOTALL,
)

_TOOL_USE_ADDENDUM = (
    "\n\n## TOOL CALL RULES\n"
    "Always invoke tools using the structured tool_calls mechanism — never embed tool calls as text.\n"
    "When a tool call is required, emit only the tool call; do not add explanatory prose alongside it.\n"
    "Never write {function: name, ...} or any text representation of a tool call."
)


def _parse_text_tool_args(kv_str: str) -> dict[str, Any]:
    """Parse args from a Hermes3 text-format tool call.

    Tries json.loads() first (handles quoted values and commas in values),
    then falls back to bare key:value comma-split.
    """
    kv_str = kv_str.strip()
    if kv_str:
        try:
            parsed = json.loads(kv_str)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    args: dict[str, Any] = {}
    for pair in kv_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, _, v = pair.partition(":")
            args[k.strip()] = v.strip()
    return args


def _parse_json_content_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Try to extract a tool call from JSON-formatted content text.

    Handles two shapes Hermes3 may emit:
      OpenAI-style: {"function": {"name": "...", "arguments": {...}}}
      Flat-style:   {"name": "...", "arguments": {...}}

    Returns (fn_name, args) or None if the text is not a recognisable tool call.
    """
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    # OpenAI-style: {"function": {"name": "...", "arguments": {...}}}
    fn = data.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if name and isinstance(name, str):
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            return name, args if isinstance(args, dict) else {}

    # Flat-style: {"name": "...", "arguments": {...}}
    name = data.get("name")
    if name and isinstance(name, str):
        args = data.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        return name, args if isinstance(args, dict) else {}

    return None


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for per-sentence TTS pipelining."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class ChatterboxTTSResponseHandler(ConversationHandler):
    """Ollama LLM + Chatterbox TTS voice output with tool dispatch."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override: str | None = startup_voice
        self._http: httpx.AsyncClient | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        self.tool_manager = BackgroundToolManager()
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Attributes referenced by LocalSTTInputMixin
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    def _mark_activity(self, label: str) -> None:
        logger.debug("Activity: %s", label)

    def copy(self) -> "ChatterboxTTSResponseHandler":
        return ChatterboxTTSResponseHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    def _load_profile_params(self) -> dict[str, str]:
        """Read profiles/<name>/chatterbox.txt as key=value pairs, if present."""
        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if not profile:
            return {}
        try:
            path = config.PROFILES_DIRECTORY / profile / "chatterbox.txt"
            if not path.exists():
                return {}
            params: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    params[k.strip()] = v.strip()
            return params
        except Exception as exc:
            logger.warning("Could not read chatterbox.txt for profile %r: %s", profile, exc)
            return {}

    @property
    def _chatterbox_url(self) -> str:
        return getattr(config, "CHATTERBOX_URL", CHATTERBOX_DEFAULT_URL)

    @property
    def _chatterbox_voice(self) -> str:
        if self._voice_override:
            return self._voice_override
        params = self._load_profile_params()
        return params.get("voice") or getattr(config, "CHATTERBOX_VOICE", CHATTERBOX_DEFAULT_VOICE)

    @property
    def _exaggeration(self) -> float:
        params = self._load_profile_params()
        return float(params.get("exaggeration", getattr(config, "CHATTERBOX_EXAGGERATION", CHATTERBOX_DEFAULT_EXAGGERATION)))

    @property
    def _cfg_weight(self) -> float:
        params = self._load_profile_params()
        return float(params.get("cfg_weight", getattr(config, "CHATTERBOX_CFG_WEIGHT", CHATTERBOX_DEFAULT_CFG_WEIGHT)))

    @property
    def _temperature(self) -> float:
        params = self._load_profile_params()
        return float(params.get("temperature", getattr(config, "CHATTERBOX_TEMPERATURE", CHATTERBOX_DEFAULT_TEMPERATURE)))

    @property
    def _gain(self) -> float:
        params = self._load_profile_params()
        return float(params.get("gain", getattr(config, "CHATTERBOX_GAIN", CHATTERBOX_DEFAULT_GAIN)))

    @property
    def _ollama_base_url(self) -> str:
        import urllib.parse
        parsed = urllib.parse.urlparse(self._chatterbox_url)
        return f"{parsed.scheme}://{parsed.hostname}:11434"

    async def _prepare_startup_credentials(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        )
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_notification])
        logger.info(
            "ChatterboxTTS handler initialised: llm=%s/api/chat tts=%s voice=%s exag=%.2f cfg=%.2f temp=%.2f",
            self._ollama_base_url,
            self._chatterbox_url,
            self._chatterbox_voice,
            self._exaggeration,
            self._cfg_weight,
            self._temperature,
        )

    async def start_up(self) -> None:
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        asyncio.create_task(self._send_startup_trigger(), name="chatterbox-startup-trigger")
        await self._stop_event.wait()

    async def _send_startup_trigger(self) -> None:
        await self._dispatch_completed_transcript("[conversation started]")

    async def shutdown(self) -> None:
        self._stop_event.set()
        await self.tool_manager.shutdown()
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive(self, frame: Any) -> None:
        """No-op: audio input is handled by LocalSTTInputMixin.receive()."""

    async def emit(self) -> Any:
        return await wait_for_item(self.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        try:
            set_custom_profile(profile)
            self._conversation_history.clear()
            return f"Applied personality {profile!r}. Conversation history reset."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        """Return predefined voice filenames from the Chatterbox server."""
        assert self._http is not None
        try:
            r = await self._http.get(f"{self._chatterbox_url}/get_predefined_voices")
            r.raise_for_status()
            return [v["filename"] for v in r.json()]
        except Exception as exc:
            logger.warning("Could not fetch Chatterbox voices: %s", exc)
            return [self._chatterbox_voice]

    def get_current_voice(self) -> str:
        return self._chatterbox_voice

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # Response cycle                                                       #
    # ------------------------------------------------------------------ #

    async def _handle_tool_notification(self, notification: ToolNotification) -> None:
        logger.info("Tool %s finished: status=%s", notification.tool_name, notification.status.value)

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """LLM → tool dispatch → TTS → PCM frames."""
        self._conversation_history.append({"role": "user", "content": transcript})

        try:
            response_text, tool_calls = await self._call_llm()
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return

        self._conversation_history.append({"role": "assistant", "content": response_text})

        if tool_calls:
            await self._dispatch_tool_calls(tool_calls)

        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": response_text})
        )

        if not response_text:
            return

        persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default"
        segments = translate(response_text, persona=persona, use_turbo=False)

        any_audio = False
        for seg in segments:
            if seg.silence_ms:
                for frame in self._pcm_to_frames(self._silence_pcm(seg.silence_ms)):
                    await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
                any_audio = True
            else:
                text = f"{seg.turbo_insert} {seg.text}" if seg.turbo_insert else seg.text
                for sentence in _split_sentences(text):
                    pcm = await self._call_chatterbox_tts(
                        sentence, exaggeration=seg.exaggeration, cfg_weight=seg.cfg_weight
                    )
                    if pcm:
                        for frame in self._pcm_to_frames(pcm):
                            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
                        any_audio = True

        if not any_audio:
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[TTS error]"})
            )

    async def _dispatch_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args = fn.get("arguments", {})
            args_json = json.dumps(args) if isinstance(args, dict) else str(args)
            call_id = uuid.uuid4().hex[:8]
            try:
                await self.tool_manager.start_tool(
                    call_id=call_id,
                    tool_call_routine=ToolCallRoutine(
                        tool_name=tool_name,
                        args_json_str=args_json,
                        deps=self.deps,
                    ),
                    is_idle_tool_call=False,
                )
                logger.info("Dispatched tool: %s (call_id=%s)", tool_name, call_id)
            except Exception as exc:
                logger.warning("Failed to dispatch tool %s: %s", tool_name, exc)

    @staticmethod
    def _trim_tool_spec(s: dict[str, Any]) -> dict[str, Any]:
        """Strip verbose parameter descriptions so the tools payload stays under ~1500 prompt tokens.

        Hermes3 8B ignores its system-prompt persona when the combined prompt exceeds ~2500
        tokens — it reverts to its baked-in "I am Hermes" identity. The play_emotion spec
        alone contributes ~1500 tokens of prose that duplicates what the system prompt already
        says. Keeping only the enum list (not the per-value prose) cuts this to <100 tokens.
        """
        import copy
        params = copy.deepcopy(s.get("parameters", {}))
        for prop in params.get("properties", {}).values():
            desc = prop.get("description", "")
            if len(desc) > 120:
                prop["description"] = desc[:120]
        return {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"][:200],
                "parameters": params,
            },
        }

    async def _call_llm(self) -> tuple[str, list[dict[str, Any]]]:
        """Call Ollama /api/chat with tool specs; returns (text, tool_calls)."""
        assert self._http is not None
        system_prompt = get_session_instructions() + _TOOL_USE_ADDENDUM
        tool_specs = get_active_tool_specs(self.deps)
        ollama_tools = [self._trim_tool_spec(s) for s in tool_specs]
        messages = [{"role": "system", "content": system_prompt}] + self._conversation_history
        logger.debug(
            "_call_llm: model=%s tools=%d sys_prompt_chars=%d sys_prompt_head=%r",
            getattr(config, "OLLAMA_MODEL", "hermes3:8b-llama3.1-q4_K_M"),
            len(ollama_tools),
            len(system_prompt),
            system_prompt[:120],
        )

        payload: dict[str, Any] = {
            "model": getattr(config, "OLLAMA_MODEL", "hermes3:8b-llama3.1-q4_K_M"),
            "messages": messages,
            "tools": ollama_tools,
            "stream": False,
        }

        nudge_attempted = False

        delay = _LLM_RETRY_BASE_DELAY
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                r = await self._http.post(
                    f"{self._ollama_base_url}/api/chat",
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                msg = data.get("message", {})
                text = (msg.get("content") or "").strip()
                tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

                # Hermes3 sometimes puts tool calls as plain text instead of tool_calls
                if not tool_calls and text:
                    m = _TEXT_TOOL_CALL_RE.match(text)
                    if m:
                        fn_name = m.group(1)
                        args = _parse_text_tool_args(m.group(2) or "")
                        tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                        logger.warning(
                            "Hermes3 text-format tool call in content field: %s(%r) — dispatching and suppressing TTS",
                            fn_name, args,
                        )
                        text = ""

                # Hermes3 may also emit JSON-format tool calls in the content field
                if not tool_calls and text:
                    json_tc = _parse_json_content_tool_call(text)
                    if json_tc is not None:
                        fn_name, args = json_tc
                        tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                        logger.warning(
                            "Hermes3 JSON-format tool call in content field: %s(%r) — dispatching and suppressing TTS",
                            fn_name, args,
                        )
                        text = ""

                if not tool_calls and not text and not nudge_attempted:
                    nudge_attempted = True
                    logger.info("Hermes3 returned empty response — attempting nudge")
                    text, tool_calls = await self._nudge_llm(messages, payload)

                return text, tool_calls
            except Exception as exc:
                if attempt == _LLM_MAX_RETRIES - 1:
                    raise
                logger.warning("LLM attempt %d/%d failed: %s: %s; retrying in %.1fs",
                               attempt + 1, _LLM_MAX_RETRIES, type(exc).__name__, exc, delay)
                await asyncio.sleep(delay)
                delay *= 2
        return "", []

    async def _nudge_llm(
        self,
        original_messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """One ephemeral retry with a tool-use nudge appended to the conversation.

        The nudge messages are not saved to _conversation_history.
        Applies the same text-format and JSON-in-content detection as _call_llm().
        If the nudge HTTP call itself fails, returns ("", []) and the turn is silently
        dropped — the caller's retry loop is not re-entered after a nudge attempt.
        """
        assert self._http is not None
        nudge_messages = original_messages + [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Please use a tool call now."},
        ]
        try:
            r = await self._http.post(
                f"{self._ollama_base_url}/api/chat",
                json={**payload, "messages": nudge_messages},
            )
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {})
            text = (msg.get("content") or "").strip()
            tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

            if not tool_calls and text:
                m = _TEXT_TOOL_CALL_RE.match(text)
                if m:
                    fn_name = m.group(1)
                    args = _parse_text_tool_args(m.group(2) or "")
                    tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                    text = ""

            if not tool_calls and text:
                json_tc = _parse_json_content_tool_call(text)
                if json_tc is not None:
                    fn_name, args = json_tc
                    tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                    text = ""

            if not tool_calls and not text:
                logger.warning("Hermes3 still empty after nudge — skipping turn")
            else:
                logger.info(
                    "Nudge recovered: text=%r tool_calls=%d",
                    text[:60] if text else "",
                    len(tool_calls),
                )
            return text, tool_calls
        except Exception as exc:
            logger.warning("Nudge LLM call failed: %s", exc)
            return "", []

    async def _call_chatterbox_tts(
        self,
        text: str,
        *,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> bytes | None:
        """POST to Chatterbox /tts in clone mode; return raw WAV bytes."""
        assert self._http is not None
        voice = self._chatterbox_voice
        ref_file = voice if voice.endswith(".wav") else f"{voice}.wav"
        payload = {
            "text": text,
            "voice_mode": "clone",
            "reference_audio_filename": ref_file,
            "output_format": "wav",
            "split_text": False,
            "exaggeration": exaggeration if exaggeration is not None else self._exaggeration,
            "cfg_weight": cfg_weight if cfg_weight is not None else self._cfg_weight,
            "temperature": self._temperature,
        }

        for attempt in range(_TTS_MAX_RETRIES):
            try:
                r = await self._http.post(f"{self._chatterbox_url}/tts", json=payload)
                r.raise_for_status()
                return self._wav_to_pcm(r.content, gain=self._gain)
            except Exception as exc:
                logger.warning("TTS attempt %d/%d failed: %s: %s", attempt + 1, _TTS_MAX_RETRIES, type(exc).__name__, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_DELAY)
        return None

    @staticmethod
    def _silence_pcm(duration_ms: int) -> bytes:
        n_samples = int(_OUTPUT_SAMPLE_RATE * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.int16).tobytes()

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes, gain: float = 1.0) -> bytes:
        """Strip WAV header, resample to 24 kHz mono int16 PCM, and apply gain."""
        import io
        import wave
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            src_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        if src_rate != _OUTPUT_SAMPLE_RATE:
            target_len = int(len(audio) * _OUTPUT_SAMPLE_RATE / src_rate)
            audio = resample(audio, target_len)
        if gain != 1.0:
            audio = audio * gain
        return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _pcm_to_frames(pcm_bytes: bytes) -> list[np.ndarray]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return [
            audio[i: i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i: i + _CHUNK_SAMPLES]) > 0
        ]


class LocalSTTChatterboxHandler(LocalSTTInputMixin, ChatterboxTTSResponseHandler):
    """Moonshine STT input + Chatterbox TTS voice output."""

    BACKEND_PROVIDER = CHATTERBOX_OUTPUT

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        # Route explicitly past LocalSTTInputMixin's OpenAI-specific override (same
        # pattern as LocalSTTGeminiTTSHandler).
        await ChatterboxTTSResponseHandler._dispatch_completed_transcript(self, transcript)
