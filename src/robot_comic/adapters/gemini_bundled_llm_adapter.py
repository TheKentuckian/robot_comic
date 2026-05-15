"""GeminiBundledLLMAdapter: expose GeminiTTSResponseHandler's LLM half as LLMBackend.

The legacy ``GeminiTTSResponseHandler._run_llm_with_tools`` (`gemini_tts.py:469`)
walks ``_LLM_MAX_TOOL_ROUNDS=5`` Gemini tool round-trips internally,
dispatches each tool inside the loop via
:func:`robot_comic.tools.core_tools.dispatch_tool_call`, and returns the
final assistant text as a ``str``. Unlike ``BaseLlamaResponseHandler._call_llm``
or ``GeminiTextResponseHandler._call_llm`` — which return
``(text, raw_tool_calls, raw_msg)`` and leave dispatch to the caller —
``_run_llm_with_tools`` is *bundled*: it owns both the LLM call and the tool
loop.

That means this adapter cannot present ``tool_calls`` to the orchestrator.
The :class:`~robot_comic.composable_pipeline.ComposablePipeline` will run
exactly one ``chat()`` round per user turn for this triple, get an
``LLMResponse(text=..., tool_calls=())``, and proceed straight to TTS. The
orchestrator's ``tool_dispatcher`` callback is **never** invoked for this
triple — tools are dispatched inside the wrapped handler against its
``self.deps``.

This is intentional and documented:

- The legacy ``LocalSTTGeminiTTSHandler`` (the wrapped handler) already
  dispatches tools internally with no orchestrator involvement, so the
  composable path behaves identically to legacy.
- Phase 4e cleanup may refactor ``GeminiTTSResponseHandler`` to expose
  ``_call_llm``-shaped LLM step so a single :class:`GeminiLLMAdapter` can
  drive both this triple and ``(moonshine, *, gemini)``. That migration is
  out of 4c.5's scope (and is its own Gemini-native API change).

## History semantics

Identical bridging pattern to :class:`~robot_comic.adapters.llama_llm_adapter.LlamaLLMAdapter`
and :class:`~robot_comic.adapters.gemini_llm_adapter.GeminiLLMAdapter`: the
legacy handler reads conversation state from ``self._conversation_history``;
the Protocol contract is "the orchestrator owns history and passes the full
list every call". The adapter swaps the handler's history with a
converted-from-messages list for the call's duration, restores on exit.

The orchestrator emits ``{"role": "user|assistant|system|tool", "content":
str}`` shapes. Gemini's API uses ``{"role": "user|model", "parts":
[{"text": ...}]}``. :func:`_orchestrator_messages_to_gemini` does the
translation:

    user      → role="user"
    assistant → role="model"
    system    → role="user" (Gemini doesn't accept "system" in history; the
                              system prompt is also passed as
                              ``system_instruction`` on the
                              ``GenerateContentConfig`` inside
                              ``_run_llm_with_tools`` via
                              ``get_session_instructions()`` — see
                              ``gemini_tts.py:480``)
    tool      → skipped (this adapter never returns ``tool_calls``, so the
                          orchestrator never appends a tool turn)
"""

from __future__ import annotations
import time
import logging
from typing import Any

from robot_comic import telemetry
from robot_comic.backends import LLMResponse
from robot_comic.adapters.gemini_tts_adapter import _GeminiTTSCompatibleHandler


logger = logging.getLogger(__name__)


def _orchestrator_messages_to_gemini(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert orchestrator-flavoured messages to Gemini's parts-shape history.

    See module docstring for the role-mapping rationale.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "user":
            out.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            out.append({"role": "model", "parts": [{"text": content}]})
        elif role == "system":
            # Gemini doesn't accept role="system" in history; prepend as a
            # user-role primer. The legacy ``_run_llm_with_tools`` also sets
            # ``GenerateContentConfig.system_instruction =
            # get_session_instructions()`` so the model already has the
            # system prompt; this primer is belt-and-braces equivalent to
            # the legacy first user turn that carries any
            # non-system_instruction system text.
            out.append({"role": "user", "parts": [{"text": content}]})
        # role == "tool" or unknown: skip — no tool history in this triple.
    return out


class GeminiBundledLLMAdapter:
    """Adapter exposing ``GeminiTTSResponseHandler`` as ``LLMBackend``.

    The wrapped handler runs LLM + tool dispatch as one bundled operation
    (``_run_llm_with_tools``). The adapter presents that as a single chat
    round with no surfaced tool calls — the orchestrator's tool loop exits
    after the first call. See module docstring for the design rationale.
    """

    def __init__(self, handler: "_GeminiTTSCompatibleHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's ``genai.Client``.

        Idempotent on the wrapped handler. Shared with
        :meth:`~robot_comic.adapters.gemini_tts_adapter.GeminiTTSAdapter.prepare`
        — same double-init pattern flagged on 4c.3 when ``GeminiLLMAdapter``
        and ``ElevenLabsTTSAdapter`` share a handler. Out of scope to fix
        here.
        """
        await self._handler._prepare_startup_credentials()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run one bundled Gemini LLM+tool round-trip and return the text response.

        ``tools`` is accepted for Protocol compliance but the legacy handler
        sources its tool list from ``self.deps`` via
        :func:`robot_comic.tools.core_tools.get_active_tool_specs`; the
        orchestrator's ``tools`` arg is dropped here. The returned
        :class:`LLMResponse` has empty ``tool_calls`` because the wrapped
        handler dispatches internally — the orchestrator's tool loop exits
        after this single call.
        """
        if tools is not None:
            logger.debug(
                "GeminiBundledLLMAdapter.chat: ignoring %d tools arg (legacy "
                "handler reads tools from deps); see module docstring",
                len(tools),
            )

        saved_history = self._handler._conversation_history
        self._handler._conversation_history = _orchestrator_messages_to_gemini(messages)
        # Lifecycle Hook #2 (#337): legacy ``GeminiTTSResponseHandler._run_llm_with_tools``
        # (``gemini_tts.py:469``) never recorded LLM duration — only its
        # ElevenLabs cousin did. The composable surface routes through this
        # adapter, so we wrap timing here to bring the bundled-Gemini triple
        # in line with the other two LLM adapters. Same attribute shape as
        # :class:`GeminiLLMAdapter`.
        _llm_start = time.perf_counter()
        try:
            text = await self._handler._run_llm_with_tools()
        finally:
            self._handler._conversation_history = saved_history
            telemetry.record_llm_duration(
                time.perf_counter() - _llm_start,
                {"gen_ai.system": "gemini", "gen_ai.operation.name": "chat"},
            )
        return LLMResponse(text=text, tool_calls=())

    async def shutdown(self) -> None:
        """No-op — ``genai.Client`` has no explicit close path here."""
        return None
