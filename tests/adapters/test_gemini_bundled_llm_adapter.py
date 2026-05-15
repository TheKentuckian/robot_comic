"""Tests for ``GeminiBundledLLMAdapter`` — Phase 4c.5 adapter wiring.

A stub LLM handler snapshots its ``_conversation_history`` at the moment
``_run_llm_with_tools`` is invoked so we can pin (a) the history-swap
behaviour and (b) the orchestrator→Gemini message-shape conversion.

The adapter intentionally returns ``LLMResponse(text=..., tool_calls=())``
even though the wrapped ``_run_llm_with_tools`` dispatches tools internally —
the orchestrator's tool-round loop is therefore never invoked for this
triple. See ``docs/superpowers/specs/2026-05-15-phase-4c5-gemini-tts-adapter.md``
Q1 for the design rationale.
"""

from __future__ import annotations
from typing import Any

import pytest

from robot_comic.backends import LLMResponse
from robot_comic.adapters.gemini_bundled_llm_adapter import GeminiBundledLLMAdapter


class _StubGeminiBundledHandler:
    """Mimics GeminiTTSResponseHandler's LLM-relevant surface.

    On every call to ``_run_llm_with_tools``, snapshots the current
    ``_conversation_history`` (so tests can assert what the LLM sees) and
    returns a canned string ``llm_return``.
    """

    def __init__(
        self,
        llm_return: str = "canned text",
        raise_exc: Exception | None = None,
    ) -> None:
        self._client: Any = object()  # non-None: _run_llm_with_tools assertion
        self._conversation_history: list[dict[str, Any]] = []
        self._llm_return = llm_return
        self._raise = raise_exc
        self.prepare_called = False
        self.history_during_call: list[dict[str, Any]] | None = None
        self.call_count = 0

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _run_llm_with_tools(self) -> str:
        self.call_count += 1
        # Snapshot what the handler sees at this moment.
        self.history_during_call = list(self._conversation_history)
        if self._raise is not None:
            raise self._raise
        return self._llm_return

    async def _call_tts_with_retry(  # pragma: no cover — required by Protocol
        self, text: str, system_instruction: str | None = None
    ) -> bytes | None:
        return None


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_calls_handler_prepare() -> None:
    handler = _StubGeminiBundledHandler()
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]
    await adapter.prepare()
    assert handler.prepare_called is True


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_llmresponse_with_text_and_no_tool_calls() -> None:
    """``chat`` returns ``LLMResponse(text=<handler-return>, tool_calls=())``."""
    handler = _StubGeminiBundledHandler(llm_return="hello world")
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]

    result = await adapter.chat([{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.text == "hello world"
    assert result.tool_calls == ()


@pytest.mark.asyncio
async def test_chat_swaps_history_for_duration_of_call() -> None:
    """While ``_run_llm_with_tools`` runs, the handler's history is the
    converted-from-messages list. After the call returns, the handler's
    original history is restored."""
    handler = _StubGeminiBundledHandler()
    handler._conversation_history = [{"role": "user", "parts": [{"text": "PRE"}]}]
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]

    await adapter.chat(
        [
            {"role": "user", "content": "from orchestrator"},
            {"role": "assistant", "content": "earlier reply"},
        ]
    )

    # During the call, the handler saw the converted messages — not the
    # pre-existing history.
    assert handler.history_during_call is not None
    assert len(handler.history_during_call) == 2
    assert handler.history_during_call[0]["role"] == "user"
    assert handler.history_during_call[0]["parts"] == [{"text": "from orchestrator"}]
    assert handler.history_during_call[1]["role"] == "model"
    assert handler.history_during_call[1]["parts"] == [{"text": "earlier reply"}]

    # After the call, the original history is restored.
    assert handler._conversation_history == [{"role": "user", "parts": [{"text": "PRE"}]}]


@pytest.mark.asyncio
async def test_chat_restores_history_on_exception() -> None:
    """If ``_run_llm_with_tools`` raises, the saved history is still restored."""
    handler = _StubGeminiBundledHandler(raise_exc=RuntimeError("llm boom"))
    handler._conversation_history = [{"role": "user", "parts": [{"text": "PRE"}]}]
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="llm boom"):
        await adapter.chat([{"role": "user", "content": "hi"}])

    assert handler._conversation_history == [{"role": "user", "parts": [{"text": "PRE"}]}]


@pytest.mark.asyncio
async def test_chat_converts_orchestrator_messages_to_gemini_shape() -> None:
    """``user`` → ``user``, ``assistant`` → ``model``, ``system`` → ``user``,
    ``tool`` → dropped."""
    handler = _StubGeminiBundledHandler()
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]

    await adapter.chat(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "tool", "tool_call_id": "abc", "name": "n", "content": "TOOL"},
            {"role": "user", "content": "U2"},
        ]
    )

    history = handler.history_during_call
    assert history is not None
    # tool turn dropped → 4 entries.
    assert len(history) == 4
    assert history[0] == {"role": "user", "parts": [{"text": "SYS"}]}
    assert history[1] == {"role": "user", "parts": [{"text": "U1"}]}
    assert history[2] == {"role": "model", "parts": [{"text": "A1"}]}
    assert history[3] == {"role": "user", "parts": [{"text": "U2"}]}


@pytest.mark.asyncio
async def test_chat_ignores_tools_arg() -> None:
    """``tools=[...]`` does not affect the call (logged-and-dropped)."""
    handler = _StubGeminiBundledHandler(llm_return="ok")
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]

    result = await adapter.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "thing"}}],
    )

    assert result.text == "ok"
    assert handler.call_count == 1


@pytest.mark.asyncio
async def test_chat_empty_history_passes_empty_list() -> None:
    """``chat([])`` results in an empty ``_conversation_history`` during the call."""
    handler = _StubGeminiBundledHandler(llm_return="empty-history reply")
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]

    result = await adapter.chat([])

    assert result.text == "empty-history reply"
    assert handler.history_during_call == []


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_is_noop() -> None:
    """``shutdown()`` does not raise and does not touch the handler."""
    handler = _StubGeminiBundledHandler()
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]
    await adapter.shutdown()
    assert handler.prepare_called is False


# ---------------------------------------------------------------------------
# telemetry — record_llm_duration (Lifecycle Hook #2, #337)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_records_llm_duration_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``chat()`` must record ``telemetry.record_llm_duration`` after each call.

    The legacy ``GeminiTTSResponseHandler._run_llm_with_tools``
    (``gemini_tts.py:469``) never recorded LLM duration — only its
    ElevenLabs cousin (``elevenlabs_tts.py:799, 829``) did. The composable
    surface routes through ``GeminiBundledLLMAdapter.chat()``; the adapter
    is the single observable LLM site post-4d default-flip, so we wrap
    timing here to bring this triple in line with the other two LLM
    adapters. Same attribute shape as ``GeminiLLMAdapter``.
    """
    from robot_comic.adapters import gemini_bundled_llm_adapter as mod

    records: list[tuple[float, dict[str, Any]]] = []

    def _recorder(duration_s: float, attrs: dict[str, Any]) -> None:
        records.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_llm_duration", _recorder)

    handler = _StubGeminiBundledHandler(llm_return="ok")
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]
    await adapter.chat([{"role": "user", "content": "hi"}])

    assert len(records) == 1, "expected exactly one telemetry record per chat() call"
    duration_s, attrs = records[0]
    assert duration_s >= 0.0
    assert attrs == {"gen_ai.system": "gemini", "gen_ai.operation.name": "chat"}


@pytest.mark.asyncio
async def test_chat_records_llm_duration_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception path still records duration — parity with the other two adapters.

    Even though legacy bundled-Gemini never recorded duration on success or
    failure, the adapter records on both paths so dashboards see consistent
    behaviour across all three LLM adapters.
    """
    from robot_comic.adapters import gemini_bundled_llm_adapter as mod

    records: list[tuple[float, dict[str, Any]]] = []

    def _recorder(duration_s: float, attrs: dict[str, Any]) -> None:
        records.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_llm_duration", _recorder)

    handler = _StubGeminiBundledHandler(raise_exc=RuntimeError("llm boom"))
    adapter = GeminiBundledLLMAdapter(handler)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="llm boom"):
        await adapter.chat([{"role": "user", "content": "hi"}])

    assert len(records) == 1, "telemetry must fire even when the LLM call raises"
    _, attrs = records[0]
    assert attrs == {"gen_ai.system": "gemini", "gen_ai.operation.name": "chat"}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_llm_backend_protocol() -> None:
    """``GeminiBundledLLMAdapter`` passes ``isinstance(LLMBackend)``."""
    from robot_comic.backends import LLMBackend

    adapter = GeminiBundledLLMAdapter(_StubGeminiBundledHandler())  # type: ignore[arg-type]
    assert isinstance(adapter, LLMBackend)
