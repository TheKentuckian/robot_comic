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
