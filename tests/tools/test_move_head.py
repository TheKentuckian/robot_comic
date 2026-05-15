from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from robot_comic.tools.move_head import MoveHead
from robot_comic.tools.core_tools import ToolDependencies


def test_move_head_schema_keeps_down_but_marks_it_explicit_only() -> None:
    tool = MoveHead()

    assert tool.parameters_schema["properties"]["direction"]["enum"] == [
        "left",
        "right",
        "up",
        "down",
        "front",
    ]
    assert "Use down only when the user explicitly asks" in tool.description
    assert "avoid it for normal conversation" in tool.parameters_schema["properties"]["direction"]["description"]


@pytest.mark.asyncio
async def test_move_head_still_allows_down() -> None:
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.reachy_mini.get_current_joint_positions.return_value = (None, (0, 0))

    result = await MoveHead()(deps, direction="down")

    assert result == {"status": "looking down"}
    deps.movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
async def test_move_head_queues_eased_goto_move() -> None:
    """Move_head opts into smoothstep easing (#264) so the head doesn't snap."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.reachy_mini.get_current_joint_positions.return_value = (None, (0, 0))

    await MoveHead()(deps, direction="left")

    deps.movement_manager.queue_move.assert_called_once()
    queued_move = deps.movement_manager.queue_move.call_args.args[0]
    assert queued_move.ease is True
