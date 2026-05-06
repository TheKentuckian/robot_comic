import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.local_stt_realtime import (
    LocalSTTOpenAIRealtimeHandler,
    LocalSTTRealtimeHandler,
)


@pytest.mark.asyncio
async def test_openai_handler_dispatch_calls_connection() -> None:
    """_dispatch_completed_transcript on the OpenAI handler sends to the realtime WebSocket."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)

    item = SimpleNamespace(create=AsyncMock())
    handler.connection = SimpleNamespace(conversation=SimpleNamespace(item=item))
    handler._safe_response_create = AsyncMock()  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("hello there")

    item.create.assert_awaited_once()
    sent = item.create.await_args.kwargs["item"]
    assert sent["role"] == "user"
    assert sent["content"][0]["text"] == "hello there"
    handler._safe_response_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_handler_dispatch_no_op_when_disconnected() -> None:
    """_dispatch_completed_transcript is silent when connection is None."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)
    handler.connection = None

    # Should not raise
    await handler._dispatch_completed_transcript("hello there")


@pytest.mark.asyncio
async def test_handle_local_stt_event_delegates_to_dispatch() -> None:
    """_handle_local_stt_event calls _dispatch_completed_transcript for completed events."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)
    handler._dispatch_completed_transcript = AsyncMock()  # type: ignore[method-assign]

    await handler._handle_local_stt_event("completed", "test transcript")

    handler._dispatch_completed_transcript.assert_awaited_once_with("test transcript")
