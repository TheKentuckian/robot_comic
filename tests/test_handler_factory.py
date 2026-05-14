"""Tests for HandlerFactory — issue #219.

Verifies that HandlerFactory.build() selects the correct handler class for every
supported (input_backend, output_backend) combination, and raises NotImplementedError
with a useful message for unsupported pairs.

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deps() -> MagicMock:
    """Return a mock ToolDependencies object."""
    return MagicMock(name="ToolDependencies")


def _build(input_backend: str, output_backend: str, deps: MagicMock) -> object:
    """Call HandlerFactory.build with sim_mode=False, no paths, no voice."""
    return HandlerFactory.build(input_backend, output_backend, deps)


# ---------------------------------------------------------------------------
# Helpers: mock a handler class so __init__ doesn't need real SDK deps
# ---------------------------------------------------------------------------


def _patch_handler(module_path: str, class_name: str):  # type: ignore[no-untyped-def]
    """Return a context manager that patches <module>.<class> with a lightweight mock."""
    return patch(f"{module_path}.{class_name}", autospec=False, new_callable=lambda: _make_mock_class(class_name))


def _make_mock_class(name: str):  # type: ignore[no-untyped-def]
    """Return a factory for a mock class that records instantiation."""

    class _MockHandler:
        _class_name = name

        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

    _MockHandler.__name__ = name
    _MockHandler.__qualname__ = name
    return lambda: _MockHandler  # new_callable must return a callable that returns the value


# Because patch(new_callable=...) is awkward here, we use patch() with new= directly.


def _patch(module_path: str, class_name: str, sentinel_cls):  # type: ignore[no-untyped-def]
    return patch(f"{module_path}.{class_name}", new=sentinel_cls)


# ---------------------------------------------------------------------------
# Supported combinations — one test per combination
# ---------------------------------------------------------------------------


class TestHandlerFactorySupportedCombinations:
    """Each supported (input, output) pair should return the expected handler type."""

    def test_hf_pair_returns_huggingface_handler(self, mock_deps: MagicMock) -> None:
        class _FakeHF:
            def __init__(self, *a, **kw):
                pass

        with patch("robot_comic.handler_factory.HuggingFaceRealtimeHandler", _FakeHF, create=True):
            with patch("robot_comic.huggingface_realtime.HuggingFaceRealtimeHandler", _FakeHF, create=True):
                # Import after patch
                import robot_comic.handler_factory as hf_mod

                original = getattr(hf_mod, "HuggingFaceRealtimeHandler", None)
                hf_mod.HuggingFaceRealtimeHandler = _FakeHF  # type: ignore[attr-defined]
                try:
                    with patch("robot_comic.huggingface_realtime.HuggingFaceRealtimeHandler", _FakeHF):
                        result = HandlerFactory.build(AUDIO_INPUT_HF, AUDIO_OUTPUT_HF, mock_deps)
                    assert isinstance(result, _FakeHF)
                finally:
                    if original is not None:
                        hf_mod.HuggingFaceRealtimeHandler = original  # type: ignore[attr-defined]
                    elif hasattr(hf_mod, "HuggingFaceRealtimeHandler"):
                        delattr(hf_mod, "HuggingFaceRealtimeHandler")

    def _assert_combo(
        self,
        input_b: str,
        output_b: str,
        deps: MagicMock,
        handler_module: str,
        handler_class: str,
    ) -> None:
        """Patch the class at import time inside HandlerFactory and verify the returned type."""

        class _FakeCls:
            def __init__(self, *a, **kw):
                pass

        _FakeCls.__name__ = handler_class

        with patch(f"robot_comic.{handler_module}.{handler_class}", _FakeCls):
            result = HandlerFactory.build(input_b, output_b, deps)

        assert isinstance(result, _FakeCls), (
            f"Expected instance of {handler_class} for ({input_b}, {output_b}), got {type(result).__name__}"
        )

    # ----- realtime pairs -----

    def test_openai_realtime_pair(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_OPENAI_REALTIME,
            AUDIO_OUTPUT_OPENAI_REALTIME,
            mock_deps,
            "openai_realtime",
            "OpenaiRealtimeHandler",
        )

    def test_gemini_live_pair(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_GEMINI_LIVE,
            AUDIO_OUTPUT_GEMINI_LIVE,
            mock_deps,
            "gemini_live",
            "GeminiLiveHandler",
        )

    # ----- moonshine pairs -----

    def test_moonshine_chatterbox(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            "chatterbox_tts",
            "LocalSTTChatterboxHandler",
        )

    def test_moonshine_gemini_tts(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_GEMINI_TTS,
            mock_deps,
            "gemini_tts",
            "LocalSTTGeminiTTSHandler",
        )

    def test_moonshine_elevenlabs(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            "elevenlabs_tts",
            "LocalSTTElevenLabsHandler",
        )

    def test_moonshine_openai_realtime_output(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_OPENAI_REALTIME,
            mock_deps,
            "local_stt_realtime",
            "LocalSTTOpenAIRealtimeHandler",
        )

    def test_moonshine_hf_output(self, mock_deps: MagicMock) -> None:
        self._assert_combo(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_HF,
            mock_deps,
            "local_stt_realtime",
            "LocalSTTHuggingFaceRealtimeHandler",
        )


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
