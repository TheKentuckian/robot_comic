"""Tests for HandlerFactory dispatch on PIPELINE_MODE (Phase 0 of pipeline refactor).

The factory now branches on PIPELINE_MODE at the top:
- ``composable`` (default) — falls through to the existing (input, output)
  selection logic for moonshine-based 3-phase pipelines.
- ``openai_realtime`` / ``gemini_live`` / ``hf_realtime`` — bundled
  speech-to-speech sessions; ignore the input/output dials.

When the ``pipeline_mode`` kwarg is omitted, the factory derives it from the
(input, output) pair (backwards-compat: existing main.py call sites and the
legacy handler_factory tests pass).
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.config import (
    AUDIO_INPUT_HF,
    AUDIO_OUTPUT_HF,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_ELEVENLABS,
    PIPELINE_MODE_COMPOSABLE,
    PIPELINE_MODE_GEMINI_LIVE,
    PIPELINE_MODE_HF_REALTIME,
    AUDIO_INPUT_OPENAI_REALTIME,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    PIPELINE_MODE_OPENAI_REALTIME,
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


# ---------------------------------------------------------------------------
# Explicit pipeline_mode kwarg drives top-level dispatch
# ---------------------------------------------------------------------------


def test_explicit_openai_realtime_returns_openai_handler(mock_deps: MagicMock) -> None:
    """``pipeline_mode=openai_realtime`` selects OpenaiRealtimeHandler regardless of input/output args."""
    fake = _fake_cls("OpenaiRealtimeHandler")
    with patch("robot_comic.openai_realtime.OpenaiRealtimeHandler", fake):
        result = HandlerFactory.build(
            # Pass irrelevant input/output to prove the mode arg wins.
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_OPENAI_REALTIME,
        )
    assert isinstance(result, fake)


def test_explicit_gemini_live_returns_gemini_live_handler(mock_deps: MagicMock) -> None:
    fake = _fake_cls("GeminiLiveHandler")
    with patch("robot_comic.gemini_live.GeminiLiveHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_GEMINI_LIVE,
        )
    assert isinstance(result, fake)


def test_explicit_hf_realtime_returns_hf_handler(mock_deps: MagicMock) -> None:
    fake = _fake_cls("HuggingFaceRealtimeHandler")
    with patch("robot_comic.huggingface_realtime.HuggingFaceRealtimeHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_HF_REALTIME,
        )
    assert isinstance(result, fake)


def test_explicit_composable_uses_input_output_selection(mock_deps: MagicMock) -> None:
    """``pipeline_mode=composable`` falls through to (input, output, llm) dispatch.

    With default LLM_BACKEND=llama, (moonshine, elevenlabs) picks the llama
    variant. The Gemini variant is exercised in test_handler_factory_gemini_llm.py.
    """
    fake = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch("robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
    assert isinstance(result, fake)


# ---------------------------------------------------------------------------
# Backwards compatibility — pipeline_mode kwarg omitted
# ---------------------------------------------------------------------------


def test_pipeline_mode_omitted_derives_from_bundled_pair(mock_deps: MagicMock) -> None:
    """Existing call sites without the kwarg still resolve bundled pairs."""
    fake = _fake_cls("OpenaiRealtimeHandler")
    with patch("robot_comic.openai_realtime.OpenaiRealtimeHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_OPENAI_REALTIME,
            AUDIO_OUTPUT_OPENAI_REALTIME,
            mock_deps,
        )
    assert isinstance(result, fake)


def test_pipeline_mode_omitted_falls_through_to_composable(mock_deps: MagicMock) -> None:
    """(moonshine, elevenlabs) without explicit mode goes to the composable branch.

    Default LLM_BACKEND=llama → LocalSTTLlamaElevenLabsHandler.
    """
    fake = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch("robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
        )
    assert isinstance(result, fake)


# ---------------------------------------------------------------------------
# Conflicts — when explicit mode contradicts input/output args
# ---------------------------------------------------------------------------


def test_bundled_mode_overrides_mismatched_input_output(mock_deps: MagicMock) -> None:
    """An explicit bundled mode wins even when input/output point elsewhere.

    This is by design: if the operator sets ``REACHY_MINI_PIPELINE_MODE=openai_realtime``
    but forgets to update AUDIO_INPUT_BACKEND/AUDIO_OUTPUT_BACKEND, we honour
    the explicit pipeline-mode declaration.
    """
    fake = _fake_cls("GeminiLiveHandler")
    with patch("robot_comic.gemini_live.GeminiLiveHandler", fake):
        # Even with HF input/output, gemini_live mode wins.
        result = HandlerFactory.build(
            AUDIO_INPUT_HF,
            AUDIO_OUTPUT_HF,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_GEMINI_LIVE,
        )
    assert isinstance(result, fake)
