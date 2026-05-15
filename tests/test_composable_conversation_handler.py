"""Tests for the Phase 4a `ComposableConversationHandler` wrapper."""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from robot_comic.conversation_handler import AudioFrame, ConversationHandler


def _make_wrapper() -> Any:
    """Build a wrapper with all collaborators mocked. Returns the wrapper instance."""
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    pipeline = MagicMock()
    pipeline.output_queue = asyncio.Queue()
    pipeline._conversation_history = []
    pipeline.start_up = AsyncMock()
    pipeline.shutdown = AsyncMock()
    pipeline.feed_audio = AsyncMock()
    pipeline.reset_history = MagicMock()

    tts_handler = MagicMock()

    def _build() -> ComposableConversationHandler:
        return _make_wrapper()

    return ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=tts_handler,
        deps=MagicMock(),
        build=_build,
    )


def test_wrapper_implements_conversation_handler_abc() -> None:
    wrapper = _make_wrapper()
    assert isinstance(wrapper, ConversationHandler)


@pytest.mark.asyncio
async def test_start_up_delegates_to_pipeline() -> None:
    wrapper = _make_wrapper()
    await wrapper.start_up()
    wrapper.pipeline.start_up.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_delegates_to_pipeline() -> None:
    wrapper = _make_wrapper()
    await wrapper.shutdown()
    wrapper.pipeline.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_forwards_to_feed_audio() -> None:
    wrapper = _make_wrapper()
    frame: AudioFrame = (16000, np.zeros(160, dtype=np.int16))
    await wrapper.receive(frame)
    wrapper.pipeline.feed_audio.assert_awaited_once_with(frame)


@pytest.mark.asyncio
async def test_emit_pulls_from_output_queue() -> None:
    wrapper = _make_wrapper()
    sentinel = (24000, np.ones(48, dtype=np.int16))
    await wrapper.pipeline.output_queue.put(sentinel)
    result = await wrapper.emit()
    assert result is sentinel


def test_get_current_voice_delegates() -> None:
    wrapper = _make_wrapper()
    wrapper._tts_handler.get_current_voice = MagicMock(return_value="Brian")
    assert wrapper.get_current_voice() == "Brian"
    wrapper._tts_handler.get_current_voice.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_voices_delegates() -> None:
    wrapper = _make_wrapper()
    wrapper._tts_handler.get_available_voices = AsyncMock(return_value=["A", "B"])
    assert await wrapper.get_available_voices() == ["A", "B"]
    wrapper._tts_handler.get_available_voices.assert_awaited_once()


@pytest.mark.asyncio
async def test_change_voice_delegates() -> None:
    wrapper = _make_wrapper()
    wrapper._tts_handler.change_voice = AsyncMock(return_value="Voice changed to X.")
    assert await wrapper.change_voice("X") == "Voice changed to X."
    wrapper._tts_handler.change_voice.assert_awaited_once_with("X")


@pytest.mark.asyncio
async def test_apply_personality_resets_history_and_reseeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _make_wrapper()
    # Pre-seed some history that should be wiped.
    wrapper.pipeline._conversation_history = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]

    # Make the MagicMock reset_history actually clear the list so the
    # post-append assertion is meaningful.
    def _real_reset(*, keep_system: bool = True) -> None:
        wrapper.pipeline._conversation_history.clear()

    wrapper.pipeline.reset_history.side_effect = _real_reset

    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.set_custom_profile",
        lambda profile: None,
    )
    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.get_session_instructions",
        lambda: "fresh instructions",
    )

    result = await wrapper.apply_personality("rodney")

    assert "Applied personality 'rodney'" in result
    wrapper.pipeline.reset_history.assert_called_once_with(keep_system=False)
    assert wrapper.pipeline._conversation_history == [
        {"role": "system", "content": "fresh instructions"},
    ]


@pytest.mark.asyncio
async def test_apply_personality_returns_failure_message_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _make_wrapper()

    def _boom(profile: str | None) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.set_custom_profile",
        _boom,
    )

    result = await wrapper.apply_personality("broken")

    assert "Failed to apply personality" in result
    assert "nope" in result
    wrapper.pipeline.reset_history.assert_not_called()


def _make_fake_pipeline() -> Any:
    p = MagicMock()
    p.output_queue = asyncio.Queue()
    p._conversation_history = []
    return p


def test_copy_returns_new_instance_from_build_closure() -> None:
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    build_count = {"n": 0}

    def _build() -> ComposableConversationHandler:
        build_count["n"] += 1
        return ComposableConversationHandler(
            pipeline=_make_fake_pipeline(),
            tts_handler=MagicMock(),
            deps=MagicMock(),
            build=_build,
        )

    original = _build()
    build_count["n"] = 0  # reset after constructing the original
    copy = original.copy()
    assert copy is not original
    assert build_count["n"] == 1


def test_copy_does_not_share_pipeline_state() -> None:
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        return ComposableConversationHandler(
            pipeline=_make_fake_pipeline(),
            tts_handler=MagicMock(),
            deps=MagicMock(),
            build=_build,
        )

    original = _build()
    copy = original.copy()

    original.pipeline._conversation_history.append({"role": "user", "content": "hi"})
    assert copy.pipeline._conversation_history == []
    assert copy.pipeline is not original.pipeline
