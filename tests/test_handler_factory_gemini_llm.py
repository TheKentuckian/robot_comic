"""Tests for HandlerFactory routing when LLM_BACKEND=gemini — issue #241 (post-Phase 4e).

Verifies that:
  - (moonshine, chatterbox) + LLM_BACKEND=gemini  → composable with
    GeminiTextChatterboxResponseHandler
  - (moonshine, elevenlabs) + LLM_BACKEND=gemini  → composable with
    GeminiTextElevenLabsResponseHandler
  - (moonshine, chatterbox) + LLM_BACKEND=llama   → composable with
    ChatterboxTTSResponseHandler (default unchanged)
  - (moonshine, <unsupported>) + LLM_BACKEND=gemini → NotImplementedError

Phase 4e (#337) retired the legacy concrete ``GeminiTextChatterboxHandler``
/ ``GeminiTextElevenLabsHandler`` classes — the factory now composes
``LocalSTTInputMixin`` over the surviving ``*ResponseHandler`` diamond
bases at construction time.

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
from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
from robot_comic.handler_factory import HandlerFactory
from robot_comic.gemini_text_handlers import (
    GeminiTextChatterboxResponseHandler,
    GeminiTextElevenLabsResponseHandler,
)
from robot_comic.composable_conversation_handler import ComposableConversationHandler


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deps() -> MagicMock:
    """Return a mock ToolDependencies object."""
    return MagicMock(name="ToolDependencies")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandlerFactoryGeminiLLM:
    """HandlerFactory should compose Gemini-text handlers when LLM_BACKEND=gemini."""

    def test_gemini_llm_chatterbox_returns_gemini_chatterbox_handler(self, mock_deps: MagicMock) -> None:
        """(moonshine, chatterbox) + LLM_BACKEND=gemini → composable Gemini+Chatterbox."""
        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = LLM_BACKEND_GEMINI
            result = HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX, mock_deps)
        assert isinstance(result, ComposableConversationHandler)
        assert isinstance(result._tts_handler, GeminiTextChatterboxResponseHandler)

    def test_gemini_llm_elevenlabs_returns_gemini_elevenlabs_handler(self, mock_deps: MagicMock) -> None:
        """(moonshine, elevenlabs) + LLM_BACKEND=gemini → composable Gemini+ElevenLabs."""
        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = LLM_BACKEND_GEMINI
            result = HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_ELEVENLABS, mock_deps)
        assert isinstance(result, ComposableConversationHandler)
        assert isinstance(result._tts_handler, GeminiTextElevenLabsResponseHandler)

    def test_llama_backend_chatterbox_returns_llama_chatterbox(self, mock_deps: MagicMock) -> None:
        """Default (LLM_BACKEND=llama) still routes to the llama-shaped composable."""
        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = LLM_BACKEND_LLAMA
            result = HandlerFactory.build(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX, mock_deps)
        assert isinstance(result, ComposableConversationHandler)
        # The llama-chatterbox builder hosts the ChatterboxTTSResponseHandler
        # directly (no Gemini-text diamond involved). Use the negative
        # assertion as well: the GeminiTextChatterboxResponseHandler is a
        # subclass of ChatterboxTTSResponseHandler, so we need both checks.
        assert isinstance(result._tts_handler, ChatterboxTTSResponseHandler)
        assert not isinstance(result._tts_handler, GeminiTextChatterboxResponseHandler)

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
