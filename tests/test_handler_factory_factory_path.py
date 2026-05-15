"""Phase 4b: dispatch behaviour under REACHY_MINI_FACTORY_PATH (#337).

Pins the routing matrix on both sides of the dial. Today's behaviour (legacy
concrete handler classes) is the default and must keep working; the new
composable path is only active when ``FACTORY_PATH=composable`` AND the
triple is ``(moonshine, llama, elevenlabs)`` — every other combo flows
through the legacy branches unchanged.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.config import (
    AUDIO_INPUT_HF,
    AUDIO_OUTPUT_HF,
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    FACTORY_PATH_LEGACY,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    FACTORY_PATH_COMPOSABLE,
    AUDIO_OUTPUT_GEMINI_LIVE,
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


def test_legacy_path_returns_legacy_handler_for_llama_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=legacy`` (default) keeps today's concrete handler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch("robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)


def test_legacy_path_returns_legacy_handler_for_llama_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=legacy`` (default) keeps today's chatterbox handler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("LocalSTTChatterboxHandler")
    with patch("robot_comic.chatterbox_tts.LocalSTTChatterboxHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)


def test_composable_path_returns_wrapper_for_llama_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=composable`` + the prod triple → ComposableConversationHandler."""
    from robot_comic import config as cfg_mod
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result.pipeline, ComposablePipeline)
    assert isinstance(result._tts_handler, fake_legacy)


def test_composable_path_wires_three_adapters(monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock) -> None:
    """All three adapters wrap the same single legacy handler instance."""
    from robot_comic import config as cfg_mod
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    pipe = result.pipeline
    assert isinstance(pipe.stt, MoonshineSTTAdapter)
    assert isinstance(pipe.llm, LlamaLLMAdapter)
    assert isinstance(pipe.tts, ElevenLabsTTSAdapter)
    # All three adapters share the same legacy handler instance.
    assert pipe.stt._handler is pipe.llm._handler
    assert pipe.llm._handler is pipe.tts._handler
    assert pipe.stt._handler is result._tts_handler


def test_composable_path_seeds_system_prompt(monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock) -> None:
    """The pipeline's system prompt is sourced from prompts.get_session_instructions."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)
    monkeypatch.setattr(
        "robot_comic.prompts.get_session_instructions",
        lambda: "TEST INSTRUCTIONS",
    )

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert result.pipeline._conversation_history[0] == {
        "role": "system",
        "content": "TEST INSTRUCTIONS",
    }


def test_composable_path_copy_constructs_fresh_legacy(monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock) -> None:
    """copy() must produce an independent wrapper + legacy handler instance."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        original = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
        copy = original.copy()

    assert copy is not original
    assert copy._tts_handler is not original._tts_handler
    assert copy.pipeline is not original.pipeline


@pytest.mark.parametrize(
    "output_backend, target_module, target_class",
    [
        (AUDIO_OUTPUT_CHATTERBOX, "robot_comic.chatterbox_tts", "LocalSTTChatterboxHandler"),
        (AUDIO_OUTPUT_GEMINI_TTS, "robot_comic.gemini_tts", "LocalSTTGeminiTTSHandler"),
        (AUDIO_OUTPUT_OPENAI_REALTIME, "robot_comic.local_stt_realtime", "LocalSTTOpenAIRealtimeHandler"),
        (AUDIO_OUTPUT_HF, "robot_comic.local_stt_realtime", "LocalSTTHuggingFaceRealtimeHandler"),
    ],
)
def test_composable_path_only_affects_llama_elevenlabs(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
    output_backend: str,
    target_module: str,
    target_class: str,
) -> None:
    """Even with FACTORY_PATH=composable, other Moonshine triples stay on legacy."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake = _fake_cls(target_class)
    with patch(f"{target_module}.{target_class}", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            output_backend,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)


@pytest.mark.parametrize(
    "pipeline_mode, input_b, output_b, target_module, target_class",
    [
        (
            PIPELINE_MODE_HF_REALTIME,
            AUDIO_INPUT_HF,
            AUDIO_OUTPUT_HF,
            "robot_comic.huggingface_realtime",
            "HuggingFaceRealtimeHandler",
        ),
        (
            PIPELINE_MODE_OPENAI_REALTIME,
            AUDIO_INPUT_OPENAI_REALTIME,
            AUDIO_OUTPUT_OPENAI_REALTIME,
            "robot_comic.openai_realtime",
            "OpenaiRealtimeHandler",
        ),
        (
            PIPELINE_MODE_GEMINI_LIVE,
            AUDIO_INPUT_GEMINI_LIVE,
            AUDIO_OUTPUT_GEMINI_LIVE,
            "robot_comic.gemini_live",
            "GeminiLiveHandler",
        ),
    ],
)
def test_composable_path_ignored_in_bundled_realtime_modes(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
    pipeline_mode: str,
    input_b: str,
    output_b: str,
    target_module: str,
    target_class: str,
) -> None:
    """Bundled-realtime modes ignore FACTORY_PATH entirely."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake = _fake_cls(target_class)
    with patch(f"{target_module}.{target_class}", fake):
        result = HandlerFactory.build(
            input_b,
            output_b,
            mock_deps,
            pipeline_mode=pipeline_mode,
        )

    assert isinstance(result, fake)


def test_composable_path_with_gemini_llm_unchanged(monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock) -> None:
    """FACTORY_PATH=composable + LLM_BACKEND=gemini stays on the legacy Gemini-text path (4c migrates this)."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake = _fake_cls("GeminiTextElevenLabsHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)
