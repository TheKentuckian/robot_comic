"""Tests for the LLM_BACKEND=llama branch in HandlerFactory (post-Phase 4e).

Before the original fix, ``LLM_BACKEND=llama`` (the default) +
``AUDIO_OUTPUT=elevenlabs`` silently returned the Gemini-hardcoded
``LocalSTTElevenLabsHandler``. The factory had a branch for
``LLM_BACKEND=gemini`` but no symmetric branch for ``llama``, so the
``LocalSTTLlamaElevenLabsHandler`` was never reachable through the factory.

This file pins the wiring: ``LLM_BACKEND=llama`` + ``elevenlabs`` output now
selects the llama-shaped composable triple. Tests for ``LLM_BACKEND=gemini``
live in ``tests/test_handler_factory_gemini_llm.py``.

Phase 4e (#337) replaced the legacy concrete-handler return values with
:class:`ComposableConversationHandler` wrappers; the underlying TTS-side
handler is :class:`LlamaElevenLabsTTSResponseHandler` (the surviving
response-handler base).
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from robot_comic.config import (
    LLM_BACKEND_LLAMA,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_ELEVENLABS,
    PIPELINE_MODE_COMPOSABLE,
)
from robot_comic.handler_factory import HandlerFactory
from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler
from robot_comic.composable_conversation_handler import ComposableConversationHandler


@pytest.fixture()
def mock_deps() -> MagicMock:
    return MagicMock(name="ToolDependencies")


def test_llama_llm_with_elevenlabs_output_selects_llama_variant(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``LLM_BACKEND=llama`` + moonshine + elevenlabs → composable llama+elevenlabs."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)

    result = HandlerFactory.build(
        AUDIO_INPUT_MOONSHINE,
        AUDIO_OUTPUT_ELEVENLABS,
        mock_deps,
        pipeline_mode=PIPELINE_MODE_COMPOSABLE,
    )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result._tts_handler, LlamaElevenLabsTTSResponseHandler)


def test_default_llm_backend_is_llama_so_default_picks_llama_variant(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """Default LLM_BACKEND is "llama" (per config.py), so an operator with no
    explicit env var still ends up on the llama path. This is the regression
    guard for the orphan-handler bug: the factory used to ignore the llama
    default and silently route to the Gemini-hardcoded handler."""
    from robot_comic import config as cfg_mod

    # Force the default — represents what an unset LLM_BACKEND env var resolves to.
    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)

    # No pipeline_mode kwarg — derivation kicks in, lands on composable.
    result = HandlerFactory.build(
        AUDIO_INPUT_MOONSHINE,
        AUDIO_OUTPUT_ELEVENLABS,
        mock_deps,
    )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result._tts_handler, LlamaElevenLabsTTSResponseHandler)
