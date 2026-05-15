"""LlamaLLMAdapter: expose BaseLlamaResponseHandler's LLM phase as LLMBackend.

The legacy ``BaseLlamaResponseHandler`` mixes the LLM call, tool dispatch,
TTS orchestration, and conversation-history management into one class. This
adapter takes a pre-constructed handler instance and presents only the
LLM-call surface (``LLMBackend.chat``) to consumers, leaving the rest
untouched.

History semantics:

    The legacy ``_call_llm`` builds its payload from
    ``self._conversation_history + extra_messages``. The Protocol contract
    is "stateless — the orchestrator owns history and passes the full list
    every call". To bridge this gap the adapter swaps the handler's
    ``_conversation_history`` to empty for the duration of the call and
    passes the operator's ``messages`` as ``extra_messages``. The original
    history is restored on the way out, even if the call raises.

Tool semantics:

    The legacy ``_stream_llm_deltas`` builds the tool list from
    ``get_active_tool_specs(self.deps)`` — the operator's tools are
    embedded in ``deps``, not passed per-call. The Protocol's ``tools``
    parameter is ignored by this adapter; it's reserved for Phase 4
    implementations that don't carry the legacy ``deps`` indirection.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any

from robot_comic.backends import ToolCall, LLMResponse


if TYPE_CHECKING:
    from robot_comic.llama_base import BaseLlamaResponseHandler


logger = logging.getLogger(__name__)


class LlamaLLMAdapter:
    """Adapter exposing ``BaseLlamaResponseHandler`` as ``LLMBackend``."""

    def __init__(self, handler: "BaseLlamaResponseHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's httpx client."""
        await self._handler._prepare_startup_credentials()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run one LLM round-trip and return the response.

        ``tools`` is accepted for Protocol compliance but the legacy handler
        sources its tool list from ``deps`` — see module docstring.
        """
        if tools is not None:
            logger.debug(
                "LlamaLLMAdapter.chat: ignoring %d tools arg (legacy handler "
                "reads tools from deps); see module docstring",
                len(tools),
            )

        saved_history = self._handler._conversation_history
        self._handler._conversation_history = []
        try:
            text, raw_tool_calls, _raw_msg = await self._handler._call_llm(
                extra_messages=messages,
            )
        finally:
            self._handler._conversation_history = saved_history

        tool_calls = tuple(_convert_tool_call(tc) for tc in raw_tool_calls)
        return LLMResponse(text=text, tool_calls=tool_calls)

    async def shutdown(self) -> None:
        """Close the underlying handler's httpx client if open."""
        # Defensive getattr in case a future handler refactor drops the
        # ``_http`` attribute. Matches the pattern in ElevenLabsTTSAdapter.
        http = getattr(self._handler, "_http", None)
        if http is not None:
            try:
                await http.aclose()
            except Exception as exc:  # pragma: no cover — best-effort cleanup
                logger.warning("LlamaLLMAdapter shutdown: aclose() raised: %s", exc)
            self._handler._http = None


def _convert_tool_call(raw: dict[str, Any]) -> ToolCall:
    """Convert a llama-server tool_call dict to the Protocol's ToolCall."""
    fn = raw.get("function", {})
    return ToolCall(
        id=str(raw.get("id", "")),
        name=str(fn.get("name", "")),
        args=fn.get("arguments") or {},
    )
