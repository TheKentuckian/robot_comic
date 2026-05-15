"""Tests for MovementManager startup motor activation."""

import time
from unittest.mock import MagicMock

import numpy as np


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
    assert call_order[0] == "enable_motors", f"enable_motors must be first; got order: {call_order[:5]}"


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


def test_maybe_log_clamp_stats_emits_summary_when_clamps_fired(caplog) -> None:
    """The periodic summary surfaces non-zero clamp activity at INFO (#272)."""
    import logging

    from robot_comic import motion_safety

    manager, _ = _make_manager()
    motion_safety.get_and_reset_clamp_stats()  # zero counters
    motion_safety._pose_clamp_counts["yaw"] = 5
    motion_safety._velocity_cap_count = 12

    with caplog.at_level(logging.INFO, logger="robot_comic.moves"):
        # loop_count == summary_interval_loops triggers the emit.
        manager._maybe_log_clamp_stats(loop_count=10, summary_interval_loops=10)

    msgs = [r.message for r in caplog.records if "motion_safety clamps" in r.message]
    assert msgs, "expected a clamp-stats summary at INFO"
    assert "yaw=5" in msgs[0]
    assert "velocity_caps=12" in msgs[0]

    # Counters reset after emit, so a second call with no new activity stays silent.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="robot_comic.moves"):
        manager._maybe_log_clamp_stats(loop_count=20, summary_interval_loops=10)
    assert not [r.message for r in caplog.records if "motion_safety clamps" in r.message]


def test_maybe_log_clamp_stats_silent_when_no_activity(caplog) -> None:
    """No clamps means no log line — keeps the journal quiet at idle."""
    import logging

    from robot_comic import motion_safety

    manager, _ = _make_manager()
    motion_safety.get_and_reset_clamp_stats()

    with caplog.at_level(logging.INFO, logger="robot_comic.moves"):
        manager._maybe_log_clamp_stats(loop_count=10, summary_interval_loops=10)

    assert not [r for r in caplog.records if "motion_safety clamps" in r.message]


def test_maybe_log_clamp_stats_includes_tracker_clamps(caplog) -> None:
    """Summary surfaces tracker-clamp activity (#308) alongside the other counters."""
    import logging

    from robot_comic import motion_safety

    manager, _ = _make_manager()
    motion_safety.get_and_reset_clamp_stats()
    motion_safety._tracker_clamp_count = 7

    with caplog.at_level(logging.INFO, logger="robot_comic.moves"):
        manager._maybe_log_clamp_stats(loop_count=10, summary_interval_loops=10)

    msgs = [r.message for r in caplog.records if "motion_safety clamps" in r.message]
    assert msgs
    assert "tracker_clamps=7" in msgs[0]


def test_maybe_log_clamp_stats_respects_interval(caplog) -> None:
    """Only every Nth tick emits — interim ticks stay silent."""
    import logging

    from robot_comic import motion_safety

    manager, _ = _make_manager()
    motion_safety.get_and_reset_clamp_stats()
    motion_safety._velocity_cap_count = 1

    with caplog.at_level(logging.INFO, logger="robot_comic.moves"):
        manager._maybe_log_clamp_stats(loop_count=5, summary_interval_loops=10)
    assert not [r for r in caplog.records if "motion_safety clamps" in r.message]
