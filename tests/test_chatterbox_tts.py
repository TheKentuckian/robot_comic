"""Tests for :class:`ChatterboxTTSResponseHandler` lifecycle.

Most chatterbox unit-test coverage lives in adjacent files
(:mod:`test_chatterbox_auto_gain`, :mod:`test_chatterbox_voice_clone`,
:mod:`test_llama_health_check`, :mod:`test_llm_warmup`). This module
captures the Phase 5e.3 idempotency guard on
:meth:`ChatterboxTTSResponseHandler._prepare_startup_credentials`
and the Phase 5e.4 sibling guard on the leaf
:class:`GeminiTextChatterboxResponseHandler` (which inherits from the
chatterbox handler but adds its own ``self._gemini_llm`` reassignment).
"""

from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


# ---------------------------------------------------------------------------
# Phase 5e.3 — _prepare_startup_credentials idempotency guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_startup_credentials_is_idempotent() -> None:
    """Second call must NOT rebuild ``_http`` or re-run the warmup chain.

    Pre-5e.3, the idempotency guard lived on
    :meth:`LocalSTTInputMixin._prepare_startup_credentials` (the host
    shell wraps the handler's prepare). Post-5e.3 the migrated triple's
    factory composes a plain handler — no shell — and the LLM and TTS
    adapters each call ``handler._prepare_startup_credentials`` once.
    Without a per-handler guard the second call leaks an extra
    ``httpx.AsyncClient`` and re-fires the llama-server health probe +
    KV-cache warmup (a streaming POST) + Chatterbox TTS warmup (a
    real TTS round-trip).
    """
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    handler = ChatterboxTTSResponseHandler(_make_deps())
    handler.tool_manager = MagicMock()
    handler._probe_llama_health = AsyncMock()  # type: ignore[method-assign]
    handler._warmup_llm_kv_cache = AsyncMock()  # type: ignore[method-assign]
    handler._warmup_tts = AsyncMock()  # type: ignore[method-assign]

    await handler._prepare_startup_credentials()
    first_http = handler._http
    assert first_http is not None
    assert handler._probe_llama_health.await_count == 1
    assert handler._warmup_llm_kv_cache.await_count == 1
    assert handler._warmup_tts.await_count == 1

    await handler._prepare_startup_credentials()
    # Same client; no leak. Warmups not re-run.
    assert handler._http is first_http
    assert handler._probe_llama_health.await_count == 1
    assert handler._warmup_llm_kv_cache.await_count == 1
    assert handler._warmup_tts.await_count == 1


# ---------------------------------------------------------------------------
# Phase 5e.4 — leaf guard on GeminiTextChatterboxResponseHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_chatterbox_prepare_startup_credentials_is_idempotent() -> None:
    """Second call must NOT re-instantiate ``GeminiLLMClient``.

    Pre-5e.4, the mixin shell's ``_prepare_startup_credentials`` gated
    the whole chain. Post-5e.4 the migrated triple's factory composes
    a plain leaf handler (no shell), and the LLM and TTS adapters each
    call ``handler._prepare_startup_credentials`` once. The inherited
    chatterbox guard (5e.3) short-circuits the chained
    ``ChatterboxTTSResponseHandler._prepare_startup_credentials`` call
    on the second invocation, but without a leaf-level guard the
    leaf-body's ``self._gemini_llm = GeminiLLMClient(...)`` reassignment
    still runs every time — leaking a fresh ``genai.Client`` per
    duplicate call.
    """
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxResponseHandler

    handler = GeminiTextChatterboxResponseHandler(_make_deps())
    handler.tool_manager = MagicMock()
    handler._probe_llama_health = AsyncMock()  # type: ignore[method-assign]
    handler._warmup_llm_kv_cache = AsyncMock()  # type: ignore[method-assign]
    handler._warmup_tts = AsyncMock()  # type: ignore[method-assign]

    with patch("robot_comic.gemini_llm.GeminiLLMClient") as mock_client_cls:
        mock_client_cls.return_value = MagicMock(name="GeminiLLMClient_instance")
        await handler._prepare_startup_credentials()
        first_client = handler._gemini_llm
        assert first_client is not None
        assert mock_client_cls.call_count == 1
        assert handler._probe_llama_health.await_count == 1
        assert handler._warmup_llm_kv_cache.await_count == 1
        assert handler._warmup_tts.await_count == 1

        await handler._prepare_startup_credentials()
        # Same Gemini client instance; no SDK-client leak.
        assert handler._gemini_llm is first_client
        assert mock_client_cls.call_count == 1
        # Inherited chatterbox guard still short-circuits its warmups.
        assert handler._probe_llama_health.await_count == 1
        assert handler._warmup_llm_kv_cache.await_count == 1
        assert handler._warmup_tts.await_count == 1
