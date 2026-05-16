"""Tests for HandlerFactory — issue #219 (post-Phase 4e rewrite).

Verifies that HandlerFactory.build() selects the correct handler shape for
every supported (input_backend, output_backend) combination, and raises
NotImplementedError with a useful message for unsupported pairs.

Phase 4e (#337) retired the FACTORY_PATH dial and deleted the legacy
``LocalSTT*Handler`` concrete classes. The composable triples now return a
:class:`ComposableConversationHandler` wrapping a host that combines
``LocalSTTInputMixin`` with the surviving ``*ResponseHandler`` base.

Bundled-realtime triples (HF / OpenAI Realtime / Gemini Live) and the
LocalSTT+realtime-output hybrids (`LocalSTTOpenAIRealtimeHandler`,
`LocalSTTHuggingFaceRealtimeHandler`) still return the legacy concrete
classes — out of scope for the composable refactor.

All handler __init__ methods are mocked so these tests have no network or SDK
dependencies.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.config import (
    AUDIO_INPUT_HF,
    AUDIO_OUTPUT_HF,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_BACKEND_ENV,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_BACKEND_ENV,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_INPUT_OPENAI_REALTIME,
    AUDIO_OUTPUT_OPENAI_REALTIME,
)
from robot_comic.handler_factory import HandlerFactory
from robot_comic.composable_conversation_handler import ComposableConversationHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deps() -> MagicMock:
    """Return a mock ToolDependencies object."""
    return MagicMock(name="ToolDependencies")


# ---------------------------------------------------------------------------
# Bundled-realtime fast paths (legacy concrete handlers)
# ---------------------------------------------------------------------------


class TestHandlerFactoryRealtimeCombinations:
    """Realtime pairs return the bundled-handler classes unchanged by 4e."""

    def _assert_realtime(
        self,
        input_b: str,
        output_b: str,
        deps: MagicMock,
        handler_module: str,
        handler_class: str,
    ) -> None:
        class _FakeCls:
            def __init__(self, *a, **kw):
                pass

        _FakeCls.__name__ = handler_class

        with patch(f"robot_comic.{handler_module}.{handler_class}", _FakeCls):
            result = HandlerFactory.build(input_b, output_b, deps)

        assert isinstance(result, _FakeCls), (
            f"Expected instance of {handler_class} for ({input_b}, {output_b}), got {type(result).__name__}"
        )

    def test_hf_pair_returns_huggingface_handler(self, mock_deps: MagicMock) -> None:
        self._assert_realtime(
            AUDIO_INPUT_HF,
            AUDIO_OUTPUT_HF,
            mock_deps,
            "huggingface_realtime",
            "HuggingFaceRealtimeHandler",
        )

    def test_openai_realtime_pair(self, mock_deps: MagicMock) -> None:
        self._assert_realtime(
            AUDIO_INPUT_OPENAI_REALTIME,
            AUDIO_OUTPUT_OPENAI_REALTIME,
            mock_deps,
            "openai_realtime",
            "OpenaiRealtimeHandler",
        )

    def test_gemini_live_pair(self, mock_deps: MagicMock) -> None:
        self._assert_realtime(
            AUDIO_INPUT_GEMINI_LIVE,
            AUDIO_OUTPUT_GEMINI_LIVE,
            mock_deps,
            "gemini_live",
            "GeminiLiveHandler",
        )

    def test_moonshine_openai_realtime_output(self, mock_deps: MagicMock) -> None:
        self._assert_realtime(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_OPENAI_REALTIME,
            mock_deps,
            "local_stt_realtime",
            "LocalSTTOpenAIRealtimeHandler",
        )

    def test_moonshine_hf_output(self, mock_deps: MagicMock) -> None:
        self._assert_realtime(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_HF,
            mock_deps,
            "local_stt_realtime",
            "LocalSTTHuggingFaceRealtimeHandler",
        )


# ---------------------------------------------------------------------------
# Composable triples — return ComposableConversationHandler
# ---------------------------------------------------------------------------


class TestHandlerFactoryComposableCombinations:
    """Each composable (moonshine, *, *) triple returns a wrapper.

    The factory builds a host that combines ``LocalSTTInputMixin`` with one
    of the surviving ``*ResponseHandler`` bases. We assert on
    ``result._tts_handler`` (the host instance the adapters wrap) to verify
    the correct host was composed.
    """

    def test_moonshine_chatterbox_default_llama_routes_to_composable(self, mock_deps: MagicMock) -> None:
        from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = "llama"
            result = HandlerFactory.build(
                AUDIO_INPUT_MOONSHINE,
                AUDIO_OUTPUT_CHATTERBOX,
                mock_deps,
            )

        assert isinstance(result, ComposableConversationHandler)
        assert isinstance(result._tts_handler, ChatterboxTTSResponseHandler)

    def test_moonshine_elevenlabs_default_llama_routes_to_composable(self, mock_deps: MagicMock) -> None:
        from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler

        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = "llama"
            result = HandlerFactory.build(
                AUDIO_INPUT_MOONSHINE,
                AUDIO_OUTPUT_ELEVENLABS,
                mock_deps,
            )

        assert isinstance(result, ComposableConversationHandler)
        assert isinstance(result._tts_handler, LlamaElevenLabsTTSResponseHandler)

    def test_moonshine_gemini_tts_routes_to_composable(self, mock_deps: MagicMock) -> None:
        from robot_comic.gemini_tts import GeminiTTSResponseHandler

        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = "llama"
            result = HandlerFactory.build(
                AUDIO_INPUT_MOONSHINE,
                AUDIO_OUTPUT_GEMINI_TTS,
                mock_deps,
            )

        assert isinstance(result, ComposableConversationHandler)
        assert isinstance(result._tts_handler, GeminiTTSResponseHandler)


# ---------------------------------------------------------------------------
# Unsupported combinations
# ---------------------------------------------------------------------------


class TestHandlerFactoryUnsupportedCombinations:
    """Unsupported pairs should raise NotImplementedError with a useful message."""

    @pytest.mark.parametrize(
        "input_b, output_b",
        [
            (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_CHATTERBOX),
            (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_ELEVENLABS),
            (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_CHATTERBOX),
            (AUDIO_INPUT_HF, AUDIO_OUTPUT_CHATTERBOX),
            (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_GEMINI_LIVE),
        ],
    )
    def test_unsupported_raises_not_implemented(
        self,
        input_b: str,
        output_b: str,
        mock_deps: MagicMock,
    ) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            HandlerFactory.build(input_b, output_b, mock_deps)

        msg = str(exc_info.value)
        # Message should mention the backend values
        assert input_b in msg, f"Error message should name the input backend {input_b!r}"
        assert output_b in msg, f"Error message should name the output backend {output_b!r}"
        # Message should point to docs
        assert "docs/audio-backends.md" in msg

    def test_unsupported_mentions_env_var_names(self, mock_deps: MagicMock) -> None:
        """The error message should name the env vars so operators know what to change."""
        with pytest.raises(NotImplementedError) as exc_info:
            HandlerFactory.build(AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_CHATTERBOX, mock_deps)

        msg = str(exc_info.value)
        assert AUDIO_INPUT_BACKEND_ENV in msg
        assert AUDIO_OUTPUT_BACKEND_ENV in msg

    def test_unknown_backends_raise_not_implemented(self, mock_deps: MagicMock) -> None:
        """Completely unknown backend strings also raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            HandlerFactory.build("bogus_input", "bogus_output", mock_deps)
