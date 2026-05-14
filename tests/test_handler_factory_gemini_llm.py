"""Tests for HandlerFactory routing when LLM_BACKEND=gemini — issue #241.

Verifies that:
  - (moonshine, chatterbox) + LLM_BACKEND=gemini  → GeminiTextChatterboxHandler
  - (moonshine, elevenlabs) + LLM_BACKEND=gemini  → GeminiTextElevenLabsHandler
  - (moonshine, chatterbox) + LLM_BACKEND=llama   → LocalSTTChatterboxHandler  (default unchanged)
  - (moonshine, <unsupported>) + LLM_BACKEND=gemini → NotImplementedError

All handler __init__ methods are mocked so these tests have zero network / SDK
dependencies.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.config import (
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_OPENAI_REALTIME,
)
from robot_comic.handler_factory import HandlerFactory


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deps() -> MagicMock:
    """Return a mock ToolDependencies object."""
    return MagicMock(name="ToolDependencies")


# ---------------------------------------------------------------------------
# Sentinel handler classes that record instantiation without SDK deps
# ---------------------------------------------------------------------------


class _FakeChatterbox:
    label = "LocalSTTChatterboxHandler"

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class _FakeGeminiChatterbox:
    label = "GeminiTextChatterboxHandler"

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class _FakeGeminiElevenLabs:
    label = "GeminiTextElevenLabsHandler"

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class _FakeElevenLabs:
    label = "LocalSTTElevenLabsHandler"

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandlerFactoryGeminiLLM:
    """HandlerFactory should route to Gemini-text handlers when LLM_BACKEND=gemini."""

    def test_gemini_llm_chatterbox_returns_gemini_chatterbox_handler(self, mock_deps: MagicMock) -> None:
        """(moonshine, chatterbox) + LLM_BACKEND=gemini → GeminiTextChatterboxHandler."""
        import robot_comic.gemini_text_handlers as _gth

        orig = _gth.GeminiTextChatterboxHandler
        try:
            _gth.GeminiTextChatterboxHandler = _FakeGeminiChatterbox  # type: ignore[attr-defined]
            with patch("robot_comic.handler_factory.config") as mock_cfg:
                mock_cfg.LLM_BACKEND = LLM_BACKEND_GEMINI
                result = HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX, mock_deps)
            assert isinstance(result, _FakeGeminiChatterbox), (
                f"Expected GeminiTextChatterboxHandler, got {type(result).__name__}"
            )
        finally:
            _gth.GeminiTextChatterboxHandler = orig  # type: ignore[attr-defined]

    def test_gemini_llm_elevenlabs_returns_gemini_elevenlabs_handler(self, mock_deps: MagicMock) -> None:
        """(moonshine, elevenlabs) + LLM_BACKEND=gemini → GeminiTextElevenLabsHandler."""
        import robot_comic.gemini_text_handlers as _gth

        orig = _gth.GeminiTextElevenLabsHandler
        try:
            _gth.GeminiTextElevenLabsHandler = _FakeGeminiElevenLabs  # type: ignore[attr-defined]
            with patch("robot_comic.handler_factory.config") as mock_cfg:
                mock_cfg.LLM_BACKEND = LLM_BACKEND_GEMINI
                result = HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_ELEVENLABS, mock_deps)
            assert isinstance(result, _FakeGeminiElevenLabs), (
                f"Expected GeminiTextElevenLabsHandler, got {type(result).__name__}"
            )
        finally:
            _gth.GeminiTextElevenLabsHandler = orig  # type: ignore[attr-defined]

    def test_llama_backend_chatterbox_returns_local_stt_chatterbox(self, mock_deps: MagicMock) -> None:
        """Default (LLM_BACKEND=llama) still routes to LocalSTTChatterboxHandler."""
        import robot_comic.chatterbox_tts as _ctt

        orig = _ctt.LocalSTTChatterboxHandler
        try:
            _ctt.LocalSTTChatterboxHandler = _FakeChatterbox  # type: ignore[attr-defined]
            with patch("robot_comic.handler_factory.config") as mock_cfg:
                mock_cfg.LLM_BACKEND = LLM_BACKEND_LLAMA
                result = HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX, mock_deps)
            assert isinstance(result, _FakeChatterbox), (
                f"Expected LocalSTTChatterboxHandler, got {type(result).__name__}"
            )
        finally:
            _ctt.LocalSTTChatterboxHandler = orig  # type: ignore[attr-defined]

    def test_gemini_llm_unsupported_output_raises_not_implemented(self, mock_deps: MagicMock) -> None:
        """(moonshine, openai_realtime_output) + LLM_BACKEND=gemini → NotImplementedError."""
        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = LLM_BACKEND_GEMINI
            with pytest.raises(NotImplementedError, match="not yet implemented"):
                HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_OPENAI_REALTIME, mock_deps)

    def test_not_implemented_message_names_supported_outputs(self, mock_deps: MagicMock) -> None:
        """The NotImplementedError message names the supported Gemini-text output backends."""
        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = LLM_BACKEND_GEMINI
            with pytest.raises(NotImplementedError) as exc_info:
                HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_OPENAI_REALTIME, mock_deps)
            msg = str(exc_info.value)
            assert AUDIO_OUTPUT_CHATTERBOX in msg
            assert AUDIO_OUTPUT_ELEVENLABS in msg
