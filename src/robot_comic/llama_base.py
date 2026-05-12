"""Shared base for llama-server LLM + local-STT response handlers.

Provides the full response cycle (LLM call, tool dispatch, TTS enqueue,
telemetry) without coupling to any specific TTS implementation.
Concrete subclasses supply _synthesize_and_enqueue() and voice management.
"""

import re
import json
import time
import uuid
import asyncio
import logging
from typing import Any, Optional

import httpx
import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from opentelemetry import trace as _otel_trace
from opentelemetry import context as _otel_context

from robot_comic import telemetry
from robot_comic.config import config
from robot_comic.prompts import get_session_instructions
from robot_comic.history_trim import trim_history_in_place
from robot_comic.tools.core_tools import ToolDependencies, get_active_tool_specs
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.tools.background_tool_manager import (
    BackgroundTool,
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0
_TOOL_RESULT_TIMEOUT: float = 5.0
# Extra seconds added after the last audio frame to cover device-buffer latency.
_ECHO_COOLDOWN_S: float = 0.5

# Split at whitespace that follows a sentence-ending punctuation mark.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for per-sentence TTS pipelining."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class BaseLlamaResponseHandler(AsyncStreamHandler, ConversationHandler):
    """llama-server LLM + tool dispatch + pluggable TTS output."""

    # Subclasses override these for telemetry labelling.
    _BACKEND_LABEL: str = "llama"
    _TTS_SYSTEM: str = "unknown"

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.sim_mode = sim_mode
        self.instance_path = instance_path
        self._voice_override: str | None = startup_voice
        self._http: httpx.AsyncClient | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        self.tool_manager = BackgroundToolManager()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        # Serialize turns so llama-server never receives two concurrent requests.
        self._turn_lock: asyncio.Lock = asyncio.Lock()

        # Attributes referenced by LocalSTTInputMixin
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None
        # OTel turn span — populated by LocalSTTInputMixin so that LLM/TTS spans
        # share the same trace as stt.infer.
        self._turn_span: Any = None
        self._turn_ctx_token: Any = None
        # Echo guard: time.perf_counter() deadline after which TTS is done playing.
        # Checked by LocalSTTInputMixin to suppress transcripts caused by speaker echo.
        self._speaking_until: float = 0.0

    def _mark_activity(self, label: str) -> None:
        logger.debug("Activity: %s", label)

    def copy(self) -> "BaseLlamaResponseHandler":
        raise NotImplementedError

    @property
    def _llama_cpp_url(self) -> str:
        from robot_comic.config import LLAMA_CPP_DEFAULT_URL

        return getattr(config, "LLAMA_CPP_URL", LLAMA_CPP_DEFAULT_URL)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def _prepare_startup_credentials(self) -> None:
        """Set up the shared HTTP client and tool manager. Subclasses should
        call super() then add their own credential / client setup.
        """
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0))
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_notification])

    async def start_up(self) -> None:
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        asyncio.create_task(self._send_startup_trigger(), name="startup-trigger")
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
        item = await wait_for_item(self.output_queue)
        if isinstance(item, tuple):
            # Update the speaking deadline: remaining queued frames + this frame + device buffer.
            remaining = self.output_queue.qsize()
            frame_s = _CHUNK_SAMPLES / _OUTPUT_SAMPLE_RATE  # 0.1 s per frame
            self._speaking_until = time.perf_counter() + (remaining + 1) * frame_s + _ECHO_COOLDOWN_S
        return item

    # ------------------------------------------------------------------ #
    # Personality / voice (common plumbing; voice specifics in subclasses)#
    # ------------------------------------------------------------------ #

    async def apply_personality(self, profile: str | None) -> str:
        from robot_comic.config import set_custom_profile

        try:
            set_custom_profile(profile)
            self._conversation_history.clear()
            return f"Applied personality {profile!r}. Conversation history reset."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        raise NotImplementedError

    def get_current_voice(self) -> str:
        raise NotImplementedError

    async def change_voice(self, voice: str) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Response cycle                                                       #
    # ------------------------------------------------------------------ #

    async def _handle_tool_notification(self, notification: ToolNotification) -> None:
        logger.info("Tool %s finished: status=%s", notification.tool_name, notification.status.value)

    async def _synthesize_and_enqueue(self, response_text: str, tts_start: float | None = None) -> None:
        raise NotImplementedError

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """LLM → tool dispatch → TTS → PCM frames (two-phase with query-tool feedback)."""
        self._conversation_history.append({"role": "user", "content": transcript})
        # Trim BEFORE building the next LLM request so unbounded sessions don't
        # blow up the context window mid-turn.
        trim_history_in_place(self._conversation_history)

        # Capture the outer span NOW — self._turn_span may be overwritten by the
        # mixin when the next STT event fires while we're blocked on the turn lock.
        # Using the local reference in the finally block avoids closing the wrong span.
        _outer_span = self._turn_span
        _reattach_token: Any = None
        if _outer_span is not None:
            _reattach_token = _otel_context.attach(_otel_trace.set_span_in_context(_outer_span))

        try:
            async with self._turn_lock:
                await self._run_turn(_outer_span)
        finally:
            if _reattach_token is not None:
                _otel_context.detach(_reattach_token)
            if _outer_span is not None:
                _outer_span.set_attribute("turn.outcome", "success")
                _outer_span.end()
                # Clear only if it still points to our span (not the next turn's).
                if self._turn_span is _outer_span:
                    self._turn_span = None

    async def _run_turn(self, outer_span: Any = None) -> None:
        _tracer = telemetry.get_tracer()
        # Stamp the outer mixin turn span so the monitor shows the right backend.
        if outer_span is not None:
            outer_span.set_attribute("robot.mode", self._BACKEND_LABEL)
            outer_span.set_attribute("gen_ai.system", "llama_cpp")
        _turn_span = _tracer.start_span(
            "turn",
            attributes={"robot.mode": self._BACKEND_LABEL, "gen_ai.system": "llama_cpp"},
        )
        _turn_start = time.perf_counter()
        _turn_ctx_token = _otel_context.attach(_otel_trace.set_span_in_context(_turn_span))
        # For startup trigger (no mixin span), tag this span directly so monitor shows "greeting".
        if outer_span is None:
            _turn_span.set_attribute("turn.excerpt", "greeting")

        _outcome = "success"
        try:
            _llm_span = _tracer.start_span(
                "llm.request",
                attributes={
                    "gen_ai.system": "llama_cpp",
                    "gen_ai.operation.name": "chat",
                },
            )
            _llm_start = time.perf_counter()
            try:
                response_text, tool_calls, raw_message = await self._call_llm()
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                _outcome = "llm_error"
                return
            finally:
                _llm_s = time.perf_counter() - _llm_start
                _llm_span.end()
                telemetry.record_llm_duration(_llm_s, {"gen_ai.system": "llama_cpp", "gen_ai.operation.name": "chat"})

            self._conversation_history.append(raw_message)

            bg_tools: list[tuple[str, BackgroundTool]] = []
            if tool_calls:
                bg_tools = await self._start_tool_calls(tool_calls)

            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))
            _tts_span = _tracer.start_span("tts.synthesize", attributes={"gen_ai.system": self._TTS_SYSTEM})
            _tts_start = time.perf_counter()
            try:
                await self._synthesize_and_enqueue(response_text, tts_start=_tts_start)
            finally:
                _tts_s = time.perf_counter() - _tts_start
                _tts_span.end()
                telemetry.record_tts(_tts_s, {"gen_ai.system": self._TTS_SYSTEM})

            if not bg_tools:
                return

            # Follow-up loop: handles multi-turn tool chains (e.g. greet → play_emotion → speak).
            # Phase-1 tools may trigger more tool calls; we keep dispatching until the model
            # produces spoken text or exhausts the depth limit.
            _MAX_FOLLOW_UP = 4
            for _phase_n in range(2, 2 + _MAX_FOLLOW_UP):
                tool_results = await self._await_tool_results(bg_tools)
                # Pure-action tools (e.g. dance) return {}; no LLM follow-up needed.
                results_with_data = {cid: res for cid, res in tool_results.items() if res}
                if not results_with_data:
                    return

                for call_id, result in results_with_data.items():
                    self._conversation_history.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": call_id,
                        }
                    )

                _llm_fu_span = _tracer.start_span(
                    "llm.request",
                    attributes={
                        "gen_ai.system": "llama_cpp",
                        "gen_ai.operation.name": "chat",
                        "llm.phase": str(_phase_n),
                    },
                )
                _llm_fu_start = time.perf_counter()
                try:
                    fu_text, fu_tool_calls, fu_raw = await self._call_llm()
                except Exception as exc:
                    logger.warning("Follow-up LLM phase %d failed: %s", _phase_n, exc)
                    _outcome = "llm_error"
                    return
                finally:
                    _llm_fu_span.end()
                    telemetry.record_llm_duration(
                        time.perf_counter() - _llm_fu_start,
                        {"gen_ai.system": "llama_cpp", "gen_ai.operation.name": "chat"},
                    )

                self._conversation_history.append(fu_raw)

                if fu_tool_calls:
                    logger.info(
                        "Follow-up phase %d: dispatching %d more tool calls",
                        _phase_n,
                        len(fu_tool_calls),
                    )
                    bg_tools = await self._start_tool_calls(fu_tool_calls)
                    if fu_text:
                        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": fu_text}))
                        _tts_fu_span = _tracer.start_span(
                            "tts.synthesize", attributes={"gen_ai.system": self._TTS_SYSTEM}
                        )
                        _tts_fu_start = time.perf_counter()
                        try:
                            await self._synthesize_and_enqueue(fu_text, tts_start=_tts_fu_start)
                        finally:
                            telemetry.record_tts(
                                time.perf_counter() - _tts_fu_start,
                                {"gen_ai.system": self._TTS_SYSTEM},
                            )
                            _tts_fu_span.end()
                    if not bg_tools:
                        return
                    continue  # await next round of tool results

                # No more tool calls — this is the final spoken response.
                if not fu_text:
                    logger.warning(
                        "Follow-up phase %d: no text and no tool calls; skipping TTS",
                        _phase_n,
                    )
                    return

                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": fu_text}))
                _tts_fu_span = _tracer.start_span("tts.synthesize", attributes={"gen_ai.system": self._TTS_SYSTEM})
                _tts_fu_start = time.perf_counter()
                try:
                    await self._synthesize_and_enqueue(fu_text, tts_start=_tts_fu_start)
                finally:
                    telemetry.record_tts(time.perf_counter() - _tts_fu_start, {"gen_ai.system": self._TTS_SYSTEM})
                    _tts_fu_span.end()
                return

            logger.warning("Follow-up tool chain exceeded max depth (%d); stopping.", _MAX_FOLLOW_UP)

        finally:
            _otel_context.detach(_turn_ctx_token)
            _turn_span.set_attribute("turn.outcome", _outcome)
            _turn_span.end()
            telemetry.record_turn(
                time.perf_counter() - _turn_start,
                {"robot.mode": self._BACKEND_LABEL, "turn.outcome": _outcome},
            )

    async def _start_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[tuple[str, BackgroundTool]]:
        """Dispatch tool calls; return (call_id, BackgroundTool) pairs."""
        results: list[tuple[str, BackgroundTool]] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args = fn.get("arguments", {})
            args_json = json.dumps(args) if isinstance(args, dict) else str(args)
            call_id = tc.get("id") or uuid.uuid4().hex[:8]
            try:
                bg_tool = await self.tool_manager.start_tool(
                    call_id=call_id,
                    tool_call_routine=ToolCallRoutine(
                        tool_name=tool_name,
                        args_json_str=args_json,
                        deps=self.deps,
                    ),
                    is_idle_tool_call=False,
                )
                results.append((call_id, bg_tool))
                logger.info("Dispatched tool: %s (call_id=%s)", tool_name, call_id)
            except Exception as exc:
                logger.warning("Failed to dispatch tool %s: %s", tool_name, exc)
        return results

    async def _await_tool_results(
        self,
        bg_tools: list[tuple[str, BackgroundTool]],
        timeout: float = _TOOL_RESULT_TIMEOUT,
    ) -> dict[str, dict[str, Any]]:
        """Await all tool tasks concurrently; return results that arrived within timeout."""

        async def _wait_one(call_id: str, bg_tool: BackgroundTool) -> tuple[str, dict[str, Any] | None]:
            if bg_tool._task is None:
                return call_id, None
            try:
                await asyncio.wait_for(asyncio.shield(bg_tool._task), timeout=timeout)
                return call_id, bg_tool.result
            except Exception:
                return call_id, None

        pairs = await asyncio.gather(*(_wait_one(cid, bt) for cid, bt in bg_tools))
        return {cid: result for cid, result in pairs if result is not None}

    async def _call_llm(
        self,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Call llama-server /v1/chat/completions; returns (text, tool_calls, raw_message).

        extra_messages are appended after conversation_history for this call only
        (not stored in history). Used to inject nudges like /no_think for Phase-2.
        """
        assert self._http is not None
        system_prompt = get_session_instructions()
        tool_specs = get_active_tool_specs(self.deps)
        messages = [{"role": "system", "content": system_prompt}] + self._conversation_history
        if extra_messages:
            messages = messages + extra_messages
        logger.info(
            "_call_llm: profile=%r tools=%d sys_chars=%d sys_head=%r",
            getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
            len(tool_specs),
            len(system_prompt),
            system_prompt[:80],
        )

        # Convert flat spec format → Chat Completions nested format:
        # {"type":"function","name":..} → {"type":"function","function":{...}}
        chat_tools = [
            {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} for t in tool_specs
        ]

        payload: dict[str, Any] = {
            "messages": messages,
            "tools": chat_tools,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        delay = _LLM_RETRY_BASE_DELAY
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                r = await self._http.post(
                    f"{self._llama_cpp_url}/v1/chat/completions",
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                raw_text = (msg.get("content") or "").strip()
                # Strip Qwen3 thinking blocks — <think>...</think> may appear
                # even when enable_thinking=false if the model emits empty ones.
                text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

                if not text and not tool_calls:
                    logger.warning(
                        "_call_llm empty response: finish_reason=%r raw=%r reasoning=%r",
                        data["choices"][0].get("finish_reason"),
                        raw_text[:300],
                        (msg.get("reasoning_content") or "")[:300],
                    )

                # Per OpenAI spec, content should be null when the message only has
                # tool_calls (empty string can confuse some model templates on next call).
                raw_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if tool_calls:
                    raw_msg["tool_calls"] = tool_calls
                return text, tool_calls, raw_msg
            except Exception as exc:
                if attempt == _LLM_MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "LLM attempt %d/%d failed: %s: %s; retrying in %.1fs",
                    attempt + 1,
                    _LLM_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
        return "", [], {"role": "assistant", "content": ""}

    # ------------------------------------------------------------------ #
    # Audio helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _silence_pcm(duration_ms: int) -> bytes:
        n_samples = int(_OUTPUT_SAMPLE_RATE * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.int16).tobytes()

    @staticmethod
    def _pcm_to_frames(pcm_bytes: bytes) -> list[np.ndarray]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return [
            audio[i : i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i : i + _CHUNK_SAMPLES]) > 0
        ]
