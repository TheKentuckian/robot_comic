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
