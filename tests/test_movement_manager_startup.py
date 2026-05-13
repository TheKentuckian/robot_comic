"""Tests for MovementManager startup motor activation."""

import threading
import time
from unittest.mock import MagicMock, call

import numpy as np
import pytest


def _make_manager():
    from robot_comic.moves import MovementManager

    robot = MagicMock()
    robot.get_current_joint_positions.return_value = (
        [0.0] * 7,
        (0.0, 0.0),
    )
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    robot.set_target.return_value = None
    robot.enable_motors.return_value = None
    return MovementManager(current_robot=robot), robot


def test_enable_motors_called_before_first_set_target() -> None:
    """enable_motors() must be called before the first set_target() in the control loop."""
    manager, robot = _make_manager()

    call_order: list[str] = []
    robot.enable_motors.side_effect = lambda *a, **kw: call_order.append("enable_motors")
    robot.set_target.side_effect = lambda *a, **kw: call_order.append("set_target")

    manager.start()
    time.sleep(0.1)  # let the loop run a few ticks
    manager._stop_event.set()
    if manager._thread:
        manager._thread.join(timeout=2.0)

    assert "enable_motors" in call_order, "enable_motors was never called"
    assert call_order[0] == "enable_motors", (
        f"enable_motors must be first; got order: {call_order[:5]}"
    )


def test_enable_motors_called_exactly_once() -> None:
    """enable_motors() should be called once at startup, not on every loop tick."""
    manager, robot = _make_manager()

    manager.start()
    time.sleep(0.15)  # enough for several loop ticks at 60 Hz
    manager._stop_event.set()
    if manager._thread:
        manager._thread.join(timeout=2.0)

    assert robot.enable_motors.call_count == 1


def test_loop_continues_if_enable_motors_raises() -> None:
    """A failure in enable_motors must not crash the control loop."""
    manager, robot = _make_manager()
    robot.enable_motors.side_effect = RuntimeError("motor fault")

    manager.start()
    time.sleep(0.1)
    manager._stop_event.set()
    if manager._thread:
        manager._thread.join(timeout=2.0)

    # Loop ran: set_target was still called despite the error
    assert robot.set_target.call_count > 0
