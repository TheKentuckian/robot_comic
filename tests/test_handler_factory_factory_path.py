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
    LLM_BACKEND_LLAMA,
    FACTORY_PATH_LEGACY,
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
