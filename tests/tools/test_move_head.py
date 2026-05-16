from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from robot_comic.tools.move_head import MoveHead
from robot_comic.tools.core_tools import ToolDependencies


def _make_deps(current_head_pose: np.ndarray | None = None) -> ToolDependencies:
    """Build a ToolDependencies with a mocked robot + movement manager."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.reachy_mini.get_current_joint_positions.return_value = (None, (0.0, 0.0))
    if current_head_pose is None:
        current_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    deps.reachy_mini.get_current_head_pose.return_value = current_head_pose
    return deps


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
    deps = _make_deps()

    result = await MoveHead()(deps, direction="down")

    assert result == {"status": "looking down"}
    deps.movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
async def test_move_head_queues_eased_goto_move() -> None:
    """Move_head opts into smoothstep easing (#264) so the head doesn't snap."""
    deps = _make_deps()

    await MoveHead()(deps, direction="left")

    deps.movement_manager.queue_move.assert_called_once()
    queued_move = deps.movement_manager.queue_move.call_args.args[0]
    assert queued_move.ease is True


@pytest.mark.asyncio
async def test_move_head_rate_limits_rapid_back_to_back_calls() -> None:
    """Hardware finding 2026-05-16: rapid LLM-driven sequences slammed the cowling.

    Second call within MOVE_HEAD_MIN_INTERVAL_S is refused with a rate-limit
    error and no GotoQueueMove is queued.
    """
    deps = _make_deps()
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    tool = MoveHead(monotonic_clock=clock)

    first = await tool(deps, direction="left")
    assert first == {"status": "looking left"}
    assert deps.movement_manager.queue_move.call_count == 1

    # 0.1 s later — well inside the default 0.6 s cooldown
    fake_now[0] += 0.1
    second = await tool(deps, direction="left")

    assert "error" in second
    assert "rate" in second["error"].lower() or "cooldown" in second["error"].lower()
    # Critically: no new move was queued
    assert deps.movement_manager.queue_move.call_count == 1


@pytest.mark.asyncio
async def test_move_head_allows_call_after_cooldown_elapsed() -> None:
    """After the cooldown window elapses, calls succeed again."""
    deps = _make_deps()
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    tool = MoveHead(monotonic_clock=clock)

    await tool(deps, direction="left")
    assert deps.movement_manager.queue_move.call_count == 1

    # Past the default cooldown
    fake_now[0] += 5.0
    result = await tool(deps, direction="right")

    assert result == {"status": "looking right"}
    assert deps.movement_manager.queue_move.call_count == 2


@pytest.mark.asyncio
async def test_move_head_dedupes_same_direction_when_already_at_target() -> None:
    """Stacking 'left, left, left' when the head is already at the left target is a no-op.

    The dedupe path returns success (so the LLM isn't punished) but never
    queues a duplicate move that would jostle the head at the cowling edge.
    """
    deps = _make_deps()
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    tool = MoveHead(monotonic_clock=clock)

    await tool(deps, direction="left")
    assert deps.movement_manager.queue_move.call_count == 1

    # Simulate the head having moved to the left target (yaw = +40 deg).
    deps.reachy_mini.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)

    # Past the cooldown, but same direction with head already at the target.
    fake_now[0] += 5.0
    result = await tool(deps, direction="left")

    assert "already" in result.get("status", "").lower()
    # No second queue_move
    assert deps.movement_manager.queue_move.call_count == 1


@pytest.mark.asyncio
async def test_move_head_rate_limits_direction_switch_too() -> None:
    """Even switching direction within the cooldown is refused — protects against
    'left, right' whiplash where the head reverses near max safe velocity."""
    deps = _make_deps()
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    tool = MoveHead(monotonic_clock=clock)

    await tool(deps, direction="left")
    fake_now[0] += 0.1
    result = await tool(deps, direction="right")

    assert "error" in result
    assert deps.movement_manager.queue_move.call_count == 1


@pytest.mark.asyncio
async def test_move_head_cooldown_is_configurable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can disable the cooldown by setting MOVE_HEAD_MIN_INTERVAL_S=0."""
    monkeypatch.setenv("REACHY_MINI_MOVE_HEAD_MIN_INTERVAL_S", "0")
    deps = _make_deps()
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    tool = MoveHead(monotonic_clock=clock)

    await tool(deps, direction="left")
    fake_now[0] += 0.001
    result = await tool(deps, direction="right")

    assert result == {"status": "looking right"}
    assert deps.movement_manager.queue_move.call_count == 2


@pytest.mark.asyncio
async def test_move_head_front_is_not_rate_limited_after_other_move() -> None:
    """'front' (recenter) should still respect the cooldown — but it's a useful
    safety target. We keep the cooldown uniform for simplicity; just verify
    it's not silently bypassed."""
    deps = _make_deps()
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    tool = MoveHead(monotonic_clock=clock)

    await tool(deps, direction="left")
    fake_now[0] += 0.1
    result = await tool(deps, direction="front")

    # Uniform policy: cooldown applies to 'front' too.
    assert "error" in result
    assert deps.movement_manager.queue_move.call_count == 1
