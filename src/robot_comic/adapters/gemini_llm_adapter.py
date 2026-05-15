"""GeminiLLMAdapter: expose GeminiTextResponseHandler's LLM phase as LLMBackend.

The legacy ``GeminiTextResponseHandler`` (``gemini_text_base.py``) overrides
``_stream_llm_deltas`` and ``_call_llm`` on top of ``BaseLlamaResponseHandler``
so the Gemini API drives the LLM step while tool dispatch, history
management, and TTS orchestration are inherited from the base class unchanged.

This adapter takes a pre-constructed handler instance (typically a concrete
``GeminiTextChatterboxHandler`` / ``GeminiTextElevenLabsHandler`` subclass)
and presents only the LLM-call surface (``LLMBackend.chat``) to consumers,
leaving the rest of the legacy machinery untouched.

History semantics:

    Identical to :class:`LlamaLLMAdapter`. The legacy ``_call_llm`` builds
    its payload from ``self._conversation_history + extra_messages``. The
    Protocol contract is "stateless — the orchestrator owns history and
    passes the full list every call". To bridge this gap the adapter swaps
    the handler's ``_conversation_history`` to empty for the duration of the
    call and passes the operator's ``messages`` as ``extra_messages``. The
    original history is restored on the way out, even if the call raises.

Tool semantics:

    ``GeminiTextResponseHandler._build_llm_messages`` builds the tool list
    from ``get_active_tool_specs(self.deps)`` — the operator's tools are
    embedded in ``deps``, not passed per-call. The Protocol's ``tools``
    parameter is ignored by this adapter; it's reserved for Phase 4
    implementations that don't carry the legacy ``deps`` indirection.

Tool-call shape:

    ``GeminiLLMClient.call_completion`` (``gemini_llm.py``) converts Gemini's
    native ``function_call`` parts into llama-server-shaped dicts *before*
    returning from ``_call_llm``: ``{"id": str, "function": {"name": str,
    "arguments": dict}}``. By the time the adapter inspects ``raw_tool_calls``
    they are in the same shape llama-server emits, so the conversion path
    here is identical to :class:`LlamaLLMAdapter`'s. The duplicate
    ``_convert_tool_call`` helper is intentional — avoiding a cross-adapter
    import keeps both modules self-contained. Phase 4e (legacy deletion) can
    consolidate the two helpers into a shared module if a third LLM adapter
    is ever added.
"""

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING, Any

from robot_comic import telemetry
from robot_comic.backends import ToolCall, LLMResponse


if TYPE_CHECKING:
    from robot_comic.gemini_text_base import GeminiTextResponseHandler


logger = logging.getLogger(__name__)


class GeminiLLMAdapter:
    """Adapter exposing ``GeminiTextResponseHandler`` as ``LLMBackend``."""

    def __init__(self, handler: "GeminiTextResponseHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's Gemini client + httpx client.

        For diamond-MRO subclasses (e.g. ``GeminiTextChatterboxResponseHandler``)
        the overridden ``_prepare_startup_credentials`` chains through the TTS
        side first, then initialises the Gemini LLM client.
        """
        await self._handler._prepare_startup_credentials()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run one Gemini LLM round-trip and return the response.

        ``tools`` is accepted for Protocol compliance but the legacy handler
        sources its tool list from ``deps`` — see module docstring.
        """
        if tools is not None:
            logger.debug(
                "GeminiLLMAdapter.chat: ignoring %d tools arg (legacy handler "
                "reads tools from deps); see module docstring",
                len(tools),
            )

        saved_history = self._handler._conversation_history
        self._handler._conversation_history = []
        # Lifecycle Hook #2 (#337): the legacy timing wrap lives in
        # ``BaseLlamaResponseHandler._run_response_loop`` (which the
        # Gemini-text handler inherits) around ``_call_llm``; the
        # composable path bypasses that loop, so we record duration here
        # to keep the histogram fed regardless of path. The legacy site
        # remains until Phase 4e cleanup. ``gen_ai.system="gemini"`` is
        # semantically correct (mirrors ``elevenlabs_tts.py:244``); the
        # legacy ``_run_response_loop`` emits ``"llama_cpp"`` by inheritance
        # accident — fixed on the new surface.
        _llm_start = time.perf_counter()
        try:
            text, raw_tool_calls, _raw_msg = await self._handler._call_llm(
                extra_messages=messages,
            )
        finally:
            self._handler._conversation_history = saved_history
            telemetry.record_llm_duration(
                time.perf_counter() - _llm_start,
                {"gen_ai.system": "gemini", "gen_ai.operation.name": "chat"},
            )

        tool_calls = tuple(_convert_tool_call(tc) for tc in raw_tool_calls)
        return LLMResponse(text=text, tool_calls=tool_calls)

    async def shutdown(self) -> None:
        """Close the underlying handler's httpx client if open."""
        # Defensive getattr in case a future handler refactor drops the
        # ``_http`` attribute. Matches the pattern in LlamaLLMAdapter.
        http = getattr(self._handler, "_http", None)
        if http is not None:
            try:
                await http.aclose()
            except Exception as exc:  # pragma: no cover — best-effort cleanup
                logger.warning("GeminiLLMAdapter shutdown: aclose() raised: %s", exc)
            self._handler._http = None


def _convert_tool_call(raw: dict[str, Any]) -> ToolCall:
    """Convert a tool_call dict to the Protocol's ToolCall.

    The dict is already in llama-server shape because the Gemini-native
    ``function_call`` → llama-server-dict conversion happens inside
    ``GeminiLLMClient.call_completion`` upstream. This helper is therefore
    byte-identical to :func:`robot_comic.adapters.llama_llm_adapter._convert_tool_call`;
    duplicated here to keep adapter modules independent.
    """
    fn = raw.get("function", {})
    return ToolCall(
        id=str(raw.get("id", "")),
        name=str(fn.get("name", "")),
        args=fn.get("arguments") or {},
    )
