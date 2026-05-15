"""Tests for the LLM_BACKEND=llama branch in HandlerFactory (Phase 0).

Before this fix, ``LLM_BACKEND=llama`` (the default) + ``AUDIO_OUTPUT=elevenlabs``
silently returned ``LocalSTTElevenLabsHandler``, which has Gemini hardcoded in
``_prepare_startup_credentials`` (``genai.Client(...)``). The factory had a
branch for ``LLM_BACKEND=gemini`` but no symmetric branch for ``llama``, so the
orphan ``LocalSTTLlamaElevenLabsHandler`` (which actually uses llama-server)
was never reachable through the factory.

This file pins the new wiring: ``LLM_BACKEND=llama`` + ``elevenlabs`` output now
selects the llama variant. Tests for ``LLM_BACKEND=gemini`` already live in
``tests/test_handler_factory_gemini_llm.py``.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.config import (
    LLM_BACKEND_LLAMA,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_ELEVENLABS,
    PIPELINE_MODE_COMPOSABLE,
)
from robot_comic.handler_factory import HandlerFactory


@pytest.fixture()
def mock_deps() -> MagicMock:
    return MagicMock(name="ToolDependencies")


def _fake_cls(name: str):
    class _Fake:
        def __init__(self, *a, **kw):
            pass

    _Fake.__name__ = name
    return _Fake


def test_llama_llm_with_elevenlabs_output_selects_llama_variant(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``LLM_BACKEND=llama`` + moonshine + elevenlabs → LocalSTTLlamaElevenLabsHandler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)

    fake = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch("robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake), f"expected LocalSTTLlamaElevenLabsHandler, got {type(result).__name__}"


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

    fake = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch("robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler", fake):
        # No pipeline_mode kwarg — derivation kicks in, lands on composable.
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
        )

    assert isinstance(result, fake)
