"""Tests for ``LlamaLLMAdapter`` — Phase 3 adapter wiring.

These tests use a tiny stub handler that records the args its ``_call_llm``
saw and returns scripted tuples, so the adapter's transform of
``(text, raw_tool_calls, raw_msg)`` → :class:`LLMResponse` is exercised
without needing a real llama-server.
"""

from __future__ import annotations
from typing import Any

import pytest

from robot_comic.backends import LLMResponse
from robot_comic.adapters.llama_llm_adapter import LlamaLLMAdapter


class _StubLlamaHandler:
    """Records ``_call_llm`` arguments and returns scripted tuples."""

    def __init__(
        self,
        responses: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self.call_args: list[list[dict[str, Any]] | None] = []
        self.history_when_called: list[list[dict[str, Any]]] = []
        self._conversation_history: list[dict[str, Any]] = []
        self._http = None
        self.prepare_called = False
        self.http_closed = False

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _call_llm(
        self,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.call_args.append(extra_messages)
        # Snapshot the handler's history at call time so the test can prove
        # the adapter cleared it.
        self.history_when_called.append(list(self._conversation_history))
        if not self._responses:
            return ("", [], {"role": "assistant", "content": ""})
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_calls_handler_prepare() -> None:
    handler = _StubLlamaHandler()
    adapter = LlamaLLMAdapter(handler)
    await adapter.prepare()
    assert handler.prepare_called is True


# ---------------------------------------------------------------------------
# chat() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_llm_response_with_text() -> None:
    handler = _StubLlamaHandler(responses=[("hello there", [], {})])
    adapter = LlamaLLMAdapter(handler)
    response = await adapter.chat([{"role": "user", "content": "hi"}])
    assert isinstance(response, LLMResponse)
    assert response.text == "hello there"
    assert response.tool_calls == ()


@pytest.mark.asyncio
async def test_chat_forwards_messages_as_extra_messages() -> None:
    handler = _StubLlamaHandler(responses=[("ok", [], {})])
    adapter = LlamaLLMAdapter(handler)
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    await adapter.chat(msgs)
    assert handler.call_args == [msgs]


@pytest.mark.asyncio
async def test_chat_clears_handler_history_for_the_call() -> None:
    """The handler's _conversation_history must be empty inside _call_llm."""
    handler = _StubLlamaHandler(responses=[("ok", [], {})])
    # Pre-populate handler history with stale state.
    handler._conversation_history = [{"role": "assistant", "content": "stale"}]
    adapter = LlamaLLMAdapter(handler)
    await adapter.chat([{"role": "user", "content": "hi"}])
    # Inside _call_llm, history was cleared so the legacy code doesn't
    # accidentally append stale history to extra_messages.
    assert handler.history_when_called == [[]]
    # And the original history is restored afterwards.
    assert handler._conversation_history == [{"role": "assistant", "content": "stale"}]


@pytest.mark.asyncio
async def test_chat_restores_history_even_when_call_raises() -> None:
    class _Boom:
        def __init__(self) -> None:
            self._conversation_history = [{"role": "user", "content": "x"}]
            self._http = None

        async def _prepare_startup_credentials(self) -> None: ...

        async def _call_llm(self, extra_messages=None):  # noqa: ANN001
            raise RuntimeError("kaboom")

    handler = _Boom()
    adapter = LlamaLLMAdapter(handler)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="kaboom"):
        await adapter.chat([{"role": "user", "content": "y"}])
    # History intact.
    assert handler._conversation_history == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# chat() — tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_converts_raw_tool_calls_to_protocol_tool_calls() -> None:
    """raw_tool_calls from _call_llm are reshaped into ``ToolCall`` instances."""
    raw_tool_calls = [
        {
            "index": 0,
            "id": "tc-123",
            "type": "function",
            "function": {"name": "dance", "arguments": {"name": "happy"}},
        },
    ]
    handler = _StubLlamaHandler(responses=[("", raw_tool_calls, {})])
    adapter = LlamaLLMAdapter(handler)
    response = await adapter.chat([{"role": "user", "content": "dance"}])
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.id == "tc-123"
    assert tc.name == "dance"
    assert tc.args == {"name": "happy"}


@pytest.mark.asyncio
async def test_chat_handles_missing_tool_call_id() -> None:
    """A tool_call missing 'id' must coerce to empty string, not crash."""
    raw_tool_calls = [
        {"function": {"name": "x", "arguments": {}}},
    ]
    handler = _StubLlamaHandler(responses=[("", raw_tool_calls, {})])
    adapter = LlamaLLMAdapter(handler)
    response = await adapter.chat([])
    assert response.tool_calls[0].id == ""
    assert response.tool_calls[0].name == "x"


@pytest.mark.asyncio
async def test_chat_handles_null_tool_call_arguments() -> None:
    """``function.arguments=None`` (rare but legal) maps to empty dict."""
    raw_tool_calls = [{"id": "tc-1", "function": {"name": "x", "arguments": None}}]
    handler = _StubLlamaHandler(responses=[("", raw_tool_calls, {})])
    adapter = LlamaLLMAdapter(handler)
    response = await adapter.chat([])
    assert response.tool_calls[0].args == {}


# ---------------------------------------------------------------------------
# tools kwarg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_accepts_and_ignores_tools_kwarg() -> None:
    """The adapter accepts ``tools=...`` for Protocol compliance but ignores
    it — the legacy handler sources tools from ``deps`` via
    ``get_active_tool_specs``. Phase 4 will pass tools through cleanly."""
    handler = _StubLlamaHandler(responses=[("ok", [], {})])
    adapter = LlamaLLMAdapter(handler)
    # Must not raise even with a tools arg.
    response = await adapter.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"name": "x"}],
    )
    assert response.text == "ok"


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_closes_handler_http() -> None:
    class _Closeable:
        def __init__(self) -> None:
            self.aclose_called = False

        async def aclose(self) -> None:
            self.aclose_called = True

    closeable = _Closeable()
    handler = _StubLlamaHandler()
    handler._http = closeable
    adapter = LlamaLLMAdapter(handler)
    await adapter.shutdown()
    assert closeable.aclose_called is True
    assert handler._http is None


@pytest.mark.asyncio
async def test_shutdown_with_no_open_http_is_safe() -> None:
    handler = _StubLlamaHandler()  # _http already None
    adapter = LlamaLLMAdapter(handler)
    await adapter.shutdown()  # must not raise
    assert handler._http is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_llm_backend_protocol() -> None:
    """``LlamaLLMAdapter`` passes isinstance(LLMBackend)."""
    from robot_comic.backends import LLMBackend

    adapter = LlamaLLMAdapter(_StubLlamaHandler())  # type: ignore[arg-type]
    assert isinstance(adapter, LLMBackend)
