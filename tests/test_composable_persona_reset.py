"""Regression tests for per-persona echo-guard reset on the composable path.

Phase 5a.1 closes the live TODO at
``composable_conversation_handler.py:172-179``. When the operator switches
personas mid-session and the wrapped TTS handler still has a non-zero
``_speaking_until`` from an in-flight or just-finished playback,
``LocalSTTInputMixin._handle_local_stt_event`` (``local_stt_realtime.py:619``)
keeps dropping the operator's next few transcripts to the new persona for
the remaining ``ECHO_COOLDOWN_MS`` window. Persona switch should be a hard
cut on listening state, mirroring the existing hard cut on conversation
history.

The audit findings and design rationale live in
``docs/superpowers/specs/2026-05-16-phase-5a1-echo-guard-persona-reset.md``.

These tests exercise the wrapper directly (no FastRTC, no live pipeline) so
the failure surface points squarely at the missing reset call.
"""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _make_pipeline_mock() -> Any:
    """Mock pipeline that mirrors the real ComposablePipeline surface the
    wrapper touches — output_queue + history + reset_history side-effect."""
    pipeline = MagicMock()
    pipeline.output_queue = asyncio.Queue()
    pipeline._conversation_history = []
    pipeline.start_up = MagicMock()
    pipeline.shutdown = MagicMock()
    pipeline.feed_audio = MagicMock()

    def _real_reset(*, keep_system: bool = True) -> None:
        pipeline._conversation_history.clear()

    pipeline.reset_history = MagicMock(side_effect=_real_reset)
    return pipeline


@pytest.mark.asyncio
async def test_apply_personality_clears_tts_handler_echo_guard_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``apply_personality`` must reset the wrapped TTS handler's
    per-session echo-guard accumulators.

    Before the Phase 5a.1 fix, the wrapper resets ``pipeline._conversation_history``
    and re-seeds the system prompt but never touches ``_tts_handler``. A
    non-zero ``_speaking_until`` from a still-playing TTS turn at the moment
    of persona switch would keep ``LocalSTTInputMixin`` suppressing the
    operator's first few transcripts to the new persona.
    """
    from robot_comic import elevenlabs_tts as mod
    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.set_custom_profile",
        lambda profile: None,
    )
    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.get_session_instructions",
        lambda: "fresh instructions",
    )

    tts_handler = ElevenLabsTTSResponseHandler(_make_deps())
    # Simulate the wrapped handler being mid-playback at the moment of switch.
    tts_handler._speaking_until = 999.0
    tts_handler._response_start_ts = 500.0
    tts_handler._response_audio_bytes = 9600

    pipeline = _make_pipeline_mock()

    def _build() -> ComposableConversationHandler:
        raise AssertionError("copy() not exercised here")

    wrapper = ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=tts_handler,
        deps=MagicMock(),
        build=_build,
    )

    result = await wrapper.apply_personality("rodney")

    assert "Applied personality 'rodney'" in result, f"apply_personality should succeed; got {result!r}"
    # The fix: per-session echo-guard accumulators on the wrapped TTS
    # handler are cleared so persona switch is a hard cut on listening
    # state. Without it, the next persona's first transcripts get
    # swallowed by the stale ``_speaking_until`` window.
    assert tts_handler._speaking_until == 0.0, (
        f"_speaking_until must be reset on persona switch; got {tts_handler._speaking_until!r}"
    )
    assert tts_handler._response_start_ts == 0.0, (
        f"_response_start_ts must be reset; got {tts_handler._response_start_ts!r}"
    )
    assert tts_handler._response_audio_bytes == 0, (
        f"_response_audio_bytes must be reset; got {tts_handler._response_audio_bytes!r}"
    )


@pytest.mark.asyncio
async def test_apply_personality_no_op_on_handler_without_echo_guard_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GeminiTTSResponseHandler`` has no echo-guard fields (no ``_speaking_until``
    etc.). The reset path must be a clean no-op on such handlers — no
    ``AttributeError`` from a blind ``setattr``, no spurious field creation.

    Pins the no-op contract so future handlers without echo guard (or a
    future refactor that removes the fields) don't crash persona switching.
    """
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.set_custom_profile",
        lambda profile: None,
    )
    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.get_session_instructions",
        lambda: "fresh instructions",
    )

    # Bare mock without the echo-guard attributes — mirrors GeminiTTSResponseHandler's
    # surface for this purpose. ``spec=[]`` would block all attribute access
    # including the wrapper's hasattr probes (which is what we want — the
    # guard must use hasattr, not bare setattr).
    tts_handler = MagicMock(spec=[])

    pipeline = _make_pipeline_mock()

    def _build() -> ComposableConversationHandler:
        raise AssertionError("copy() not exercised here")

    wrapper = ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=tts_handler,
        deps=MagicMock(),
        build=_build,
    )

    # Must not raise. Must succeed.
    result = await wrapper.apply_personality("rodney")
    assert "Applied personality 'rodney'" in result, (
        f"apply_personality should still succeed on handlers without echo-guard state; got {result!r}"
    )
    # The handler must not have grown the fields (defensive — verifies the
    # guard is hasattr-based, not blind setattr).
    assert not hasattr(tts_handler, "_speaking_until"), (
        "Reset must not create _speaking_until on a handler that did not have it"
    )
