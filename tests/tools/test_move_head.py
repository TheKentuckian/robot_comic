from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.tools.move_head import MoveHead


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
