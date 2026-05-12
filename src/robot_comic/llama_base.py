"""Shared base for llama-server LLM + local-STT response handlers.

Provides the full response cycle (LLM call, tool dispatch, TTS enqueue,
telemetry) without coupling to any specific TTS implementation.
Concrete subclasses supply _synthesize_and_enqueue() and voice management.
"""

import json
import re
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
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.prompts import get_session_instructions
from robot_comic.tools.core_tools import ToolDependencies, get_active_tool_specs
from robot_comic.tools.background_tool_manager import (
    BackgroundTool,
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)

logger = logging.getLogger(__name__)

_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400           # 100 ms at 24 kHz
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0
_TOOL_RESULT_TIMEOUT: float = 5.0


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

    def _mark_activity(self, label: str) -> None:
        logger.debug("Activity: %s", label)

    def _close_turn_span(self, outcome: str) -> None:
        """Close the OTel turn span opened by LocalSTTInputMixin.

        Note: _turn_ctx_token holds the mixin's stale token from a different asyncio
        task — we do NOT detach it here (causes "Failed to detach context" errors).
        The re-attach token created in _dispatch_completed_transcript is detached there.
        """
        self._turn_ctx_token = None
        if self._turn_span is not None:
            self._turn_span.set_attribute("turn.outcome", outcome)
            self._turn_span.end()
            self._turn_span = None

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
        call super() then add their own credential / client setup."""
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        )
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
        return await wait_for_item(self.output_queue)

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

        # Re-attach the OTel context from the mixin's turn span (set by
        # LocalSTTInputMixin) so that llm.request and tts.synthesize spans
        # land in the same trace as stt.infer — making the monitor display them.
        _outer_span = self._turn_span
        _reattach_token: Any = None
        if _outer_span is not None:
            _reattach_token = _otel_context.attach(_otel_trace.set_span_in_context(_outer_span))

        try:
            async with self._turn_lock:
                await self._run_turn()
        finally:
            if _reattach_token is not None:
                _otel_context.detach(_reattach_token)
            if self._turn_span is not None:
                self._close_turn_span("success")

    async def _run_turn(self) -> None:
        _tracer = telemetry.get_tracer()
        # Stamp the outer mixin turn span with the actual backend so the monitor
        # shows the right mode (e.g. "llama_gemini_tts" instead of "local_stt").
        if self._turn_span is not None:
            self._turn_span.set_attribute("robot.mode", self._BACKEND_LABEL)
            self._turn_span.set_attribute("gen_ai.system", "llama_cpp")
        _turn_span = _tracer.start_span(
            "turn",
            attributes={"robot.mode": self._BACKEND_LABEL, "gen_ai.system": "llama_cpp"},
        )
        _turn_start = time.perf_counter()
        _turn_ctx_token = _otel_context.attach(_otel_trace.set_span_in_context(_turn_span))
        # For startup trigger (no mixin span), tag this span directly so monitor shows "greeting".
        if self._turn_span is None:
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
                telemetry.record_llm_duration(
                    _llm_s, {"gen_ai.system": "llama_cpp", "gen_ai.operation.name": "chat"}
                )

            self._conversation_history.append(raw_message)

            bg_tools: list[tuple[str, BackgroundTool]] = []
            if tool_calls:
                bg_tools = await self._start_tool_calls(tool_calls)

            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": response_text})
            )
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

            tool_results = await self._await_tool_results(bg_tools)
            # Only feed non-empty results back to the LLM; pure action tools
            # (e.g. dance) return {} and need no follow-up.
            results_with_data = {cid: res for cid, res in tool_results.items() if res}
            if not results_with_data:
                return

            for call_id, result in results_with_data.items():
                self._conversation_history.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": call_id,
                })

            _llm2_span = _tracer.start_span(
                "llm.request",
                attributes={
                    "gen_ai.system": "llama_cpp",
                    "gen_ai.operation.name": "chat",
                    "llm.phase": "2",
                },
            )
            _llm2_start = time.perf_counter()
            try:
                follow_up_text, _, _ = await self._call_llm()
            except Exception as exc:
                logger.warning("Phase-2 LLM call failed: %s", exc)
                _outcome = "llm_error"
                return
            finally:
                _llm2_span.end()
                telemetry.record_llm_duration(
                    time.perf_counter() - _llm2_start,
                    {"gen_ai.system": "llama_cpp", "gen_ai.operation.name": "chat"},
                )

            if not follow_up_text:
                logger.warning("Phase-2 LLM returned empty text (Qwen3 think-only?); skipping TTS")
                return
            self._conversation_history.append({"role": "assistant", "content": follow_up_text})
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": follow_up_text})
            )
            _tts2_span = _tracer.start_span("tts.synthesize", attributes={"gen_ai.system": self._TTS_SYSTEM})
            _tts2_start = time.perf_counter()
            try:
                await self._synthesize_and_enqueue(follow_up_text, tts_start=_tts2_start)
            finally:
                telemetry.record_tts(
                    time.perf_counter() - _tts2_start, {"gen_ai.system": self._TTS_SYSTEM}
                )
                _tts2_span.end()

        finally:
            _otel_context.detach(_turn_ctx_token)
            _turn_span.set_attribute("turn.outcome", _outcome)
            _turn_span.end()
            telemetry.record_turn(
                time.perf_counter() - _turn_start,
                {"robot.mode": self._BACKEND_LABEL, "turn.outcome": _outcome},
            )

    async def _start_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[tuple[str, BackgroundTool]]:
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
        async def _wait_one(
            call_id: str, bg_tool: BackgroundTool
        ) -> tuple[str, dict[str, Any] | None]:
            if bg_tool._task is None:
                return call_id, None
            try:
                await asyncio.wait_for(asyncio.shield(bg_tool._task), timeout=timeout)
                return call_id, bg_tool.result
            except Exception:
                return call_id, None

        pairs = await asyncio.gather(*(_wait_one(cid, bt) for cid, bt in bg_tools))
        return {cid: result for cid, result in pairs if result is not None}

    async def _call_llm(self) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Call llama-server /v1/chat/completions; returns (text, tool_calls, raw_message)."""
        assert self._http is not None
        system_prompt = get_session_instructions()
        tool_specs = get_active_tool_specs(self.deps)
        messages = [{"role": "system", "content": system_prompt}] + self._conversation_history
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
            {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
            for t in tool_specs
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

                raw_msg: dict[str, Any] = {"role": "assistant", "content": text}
                if tool_calls:
                    raw_msg["tool_calls"] = tool_calls
                return text, tool_calls, raw_msg
            except Exception as exc:
                if attempt == _LLM_MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "LLM attempt %d/%d failed: %s: %s; retrying in %.1fs",
                    attempt + 1, _LLM_MAX_RETRIES, type(exc).__name__, exc, delay,
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
            audio[i: i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i: i + _CHUNK_SAMPLES]) > 0
        ]
