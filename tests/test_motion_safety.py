"""Unit tests for robot_comic.motion_safety — clamp and velocity-cap helpers.

All tests are pure-math: no robot hardware, no SDK I/O, no external services.
"""

from __future__ import annotations
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_comic.motion_safety import (
    HEAD_YAW_MAX_RAD,
    HEAD_YAW_MIN_RAD,
    HEAD_ROLL_MAX_RAD,
    HEAD_ROLL_MIN_RAD,
    HEAD_PITCH_MAX_RAD,
    HEAD_PITCH_MIN_RAD,
    _step_toward,
    clamp_head_pose,
    cap_head_velocity,
    _clamped_axes_seen,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pose(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """Build a 4×4 SE3 head-pose matrix from RPY angles (degrees)."""
    rot = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.as_matrix()
    return mat


def _extract_rpy_deg(pose: np.ndarray) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) in degrees from a 4×4 pose matrix."""
    r = Rotation.from_matrix(pose[:3, :3])
    roll, pitch, yaw = r.as_euler("xyz", degrees=True)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# clamp_head_pose — within-envelope passes through unchanged
# ---------------------------------------------------------------------------


class TestClampHeadPoseInEnvelope:
    """Poses within the safe envelope should pass through without modification."""

    def test_neutral_pose_unchanged(self) -> None:
        pose = _make_pose(0.0, 0.0, 0.0)
        result = clamp_head_pose(pose)
        np.testing.assert_array_almost_equal(result, pose)

    def test_small_yaw_unchanged(self) -> None:
        yaw_deg = math.degrees(HEAD_YAW_MAX_RAD) * 0.5
        pose = _make_pose(yaw=yaw_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(yaw - yaw_deg) < 0.01

    def test_small_pitch_unchanged(self) -> None:
        pitch_deg = math.degrees(HEAD_PITCH_MAX_RAD) * 0.5
        pose = _make_pose(pitch=pitch_deg)
        result = clamp_head_pose(pose)
        _, pitch, _ = _extract_rpy_deg(result)
        assert abs(pitch - pitch_deg) < 0.01

    def test_small_roll_unchanged(self) -> None:
        roll_deg = math.degrees(HEAD_ROLL_MAX_RAD) * 0.5
        pose = _make_pose(roll=roll_deg)
        result = clamp_head_pose(pose)
        roll, _, _ = _extract_rpy_deg(result)
        assert abs(roll - roll_deg) < 0.01

    def test_combined_within_envelope_unchanged(self) -> None:
        pose = _make_pose(roll=10.0, pitch=15.0, yaw=-20.0)
        result = clamp_head_pose(pose)
        np.testing.assert_array_almost_equal(result[:3, :3], pose[:3, :3], decimal=6)


# ---------------------------------------------------------------------------
# clamp_head_pose — outside-envelope is clamped; other axes pass through
# ---------------------------------------------------------------------------


class TestClampHeadPoseOutsideEnvelope:
    """When one axis exceeds the limit the clamp brings it to the boundary;
    the other two axes must remain unchanged.
    """

    def test_yaw_exceeds_max_is_clamped(self) -> None:
        _clamped_axes_seen.discard("yaw")
        exceed_deg = math.degrees(HEAD_YAW_MAX_RAD) + 15.0
        pose = _make_pose(yaw=exceed_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(yaw - math.degrees(HEAD_YAW_MAX_RAD)) < 0.5
        assert abs(roll) < 0.01
        assert abs(pitch) < 0.01

    def test_yaw_below_min_is_clamped(self) -> None:
        _clamped_axes_seen.discard("yaw")
        exceed_deg = math.degrees(HEAD_YAW_MIN_RAD) - 15.0
        pose = _make_pose(yaw=exceed_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(yaw - math.degrees(HEAD_YAW_MIN_RAD)) < 0.5
        assert abs(roll) < 0.01
        assert abs(pitch) < 0.01

    def test_pitch_exceeds_max_is_clamped(self) -> None:
        _clamped_axes_seen.discard("pitch")
        exceed_deg = math.degrees(HEAD_PITCH_MAX_RAD) + 10.0
        pose = _make_pose(pitch=exceed_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(pitch - math.degrees(HEAD_PITCH_MAX_RAD)) < 0.5
        assert abs(roll) < 0.01
        assert abs(yaw) < 0.01

    def test_pitch_below_min_is_clamped(self) -> None:
        _clamped_axes_seen.discard("pitch")
        exceed_deg = math.degrees(HEAD_PITCH_MIN_RAD) - 10.0
        pose = _make_pose(pitch=exceed_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(pitch - math.degrees(HEAD_PITCH_MIN_RAD)) < 0.5

    def test_roll_exceeds_max_is_clamped(self) -> None:
        _clamped_axes_seen.discard("roll")
        exceed_deg = math.degrees(HEAD_ROLL_MAX_RAD) + 20.0
        pose = _make_pose(roll=exceed_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(roll - math.degrees(HEAD_ROLL_MAX_RAD)) < 0.5
        assert abs(pitch) < 0.01
        assert abs(yaw) < 0.01

    def test_roll_below_min_is_clamped(self) -> None:
        _clamped_axes_seen.discard("roll")
        exceed_deg = math.degrees(HEAD_ROLL_MIN_RAD) - 20.0
        pose = _make_pose(roll=exceed_deg)
        result = clamp_head_pose(pose)
        roll, pitch, yaw = _extract_rpy_deg(result)
        assert abs(roll - math.degrees(HEAD_ROLL_MIN_RAD)) < 0.5

    def test_only_offending_axis_is_changed(self) -> None:
        """Yaw clamping must not disturb pitch or roll."""
        _clamped_axes_seen.discard("yaw")
        pose = _make_pose(roll=5.0, pitch=10.0, yaw=math.degrees(HEAD_YAW_MAX_RAD) + 20.0)
        result = clamp_head_pose(pose)
        roll_r, pitch_r, yaw_r = _extract_rpy_deg(result)
        assert abs(roll_r - 5.0) < 0.5
        assert abs(pitch_r - 10.0) < 0.5
        assert abs(yaw_r - math.degrees(HEAD_YAW_MAX_RAD)) < 0.5


# ---------------------------------------------------------------------------
# cap_head_velocity — step is bounded by max_vel * dt
# ---------------------------------------------------------------------------


class TestCapHeadVelocity:
    """Per-tick angular advance must not exceed max_vel * dt on any axis."""

    MAX_VEL = 1.5  # rad/s (default)
    DT = 1.0 / 60.0  # ~60 Hz

    def test_large_yaw_step_is_bounded(self) -> None:
        current = _make_pose(yaw=0.0)
        # Target is 90 deg away — far beyond one tick's budget
        target = _make_pose(yaw=90.0)
        result = cap_head_velocity(current, target, self.DT, self.MAX_VEL)
        _, _, yaw_r = _extract_rpy_deg(result)
        max_step_deg = math.degrees(self.MAX_VEL * self.DT)
        # Should have advanced at most max_step_deg from 0
        assert yaw_r <= max_step_deg + 0.01
        assert yaw_r > 0.0  # must advance toward target

    def test_large_pitch_step_is_bounded(self) -> None:
        current = _make_pose(pitch=0.0)
        target = _make_pose(pitch=45.0)
        result = cap_head_velocity(current, target, self.DT, self.MAX_VEL)
        _, pitch_r, _ = _extract_rpy_deg(result)
        max_step_deg = math.degrees(self.MAX_VEL * self.DT)
        assert pitch_r <= max_step_deg + 0.01
        assert pitch_r > 0.0

    def test_small_step_reaches_target(self) -> None:
        """A step already within the budget must not be further damped."""
        max_step_rad = self.MAX_VEL * self.DT
        tiny_yaw_deg = math.degrees(max_step_rad * 0.5)
        current = _make_pose(yaw=0.0)
        target = _make_pose(yaw=tiny_yaw_deg)
        result = cap_head_velocity(current, target, self.DT, self.MAX_VEL)
        _, _, yaw_r = _extract_rpy_deg(result)
        # Should reach the target exactly
        assert abs(yaw_r - tiny_yaw_deg) < 0.01

    def test_at_target_no_change(self) -> None:
        """When already at the target, output equals target."""
        pose = _make_pose(roll=5.0, pitch=-10.0, yaw=20.0)
        result = cap_head_velocity(pose, pose, self.DT, self.MAX_VEL)
        np.testing.assert_array_almost_equal(result, pose)

    def test_zero_dt_returns_target(self) -> None:
        """With dt=0 the function must not divide by zero and should return target."""
        current = _make_pose(yaw=0.0)
        target = _make_pose(yaw=45.0)
        result = cap_head_velocity(current, target, 0.0, self.MAX_VEL)
        np.testing.assert_array_almost_equal(result, target)

    def test_negative_direction_bounded(self) -> None:
        """Velocity cap also applies to motion in the negative direction."""
        current = _make_pose(yaw=0.0)
        target = _make_pose(yaw=-90.0)
        result = cap_head_velocity(current, target, self.DT, self.MAX_VEL)
        _, _, yaw_r = _extract_rpy_deg(result)
        max_step_deg = math.degrees(self.MAX_VEL * self.DT)
        assert yaw_r >= -(max_step_deg + 0.01)
        assert yaw_r < 0.0  # must advance in the negative direction

    def test_multi_tick_convergence(self) -> None:
        """Running cap_head_velocity for enough ticks eventually reaches the target."""
        current = _make_pose(yaw=0.0)
        target = _make_pose(yaw=30.0)
        pose = current
        for _ in range(1000):  # 1000 ticks @ 60 Hz → ~16.7 s
            pose = cap_head_velocity(pose, target, self.DT, self.MAX_VEL)
        _, _, yaw_final = _extract_rpy_deg(pose)
        assert abs(yaw_final - 30.0) < 0.1


# ---------------------------------------------------------------------------
# _step_toward — internal helper
# ---------------------------------------------------------------------------


class TestStepToward:
    def test_reaches_target_when_within_budget(self) -> None:
        assert _step_toward(0.0, 0.5, 1.0) == pytest.approx(0.5)

    def test_capped_when_exceeds_budget(self) -> None:
        assert _step_toward(0.0, 2.0, 1.0) == pytest.approx(1.0)

    def test_negative_direction(self) -> None:
        assert _step_toward(0.0, -2.0, 1.0) == pytest.approx(-1.0)

    def test_zero_step(self) -> None:
        assert _step_toward(5.0, 5.0, 1.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# clamp_head_pose — cheap allocation check
# ---------------------------------------------------------------------------


class TestClampAllocation:
    """clamp_head_pose should not GC-thrash: the in-envelope fast path returns
    the original object, and the clamped path does a single copy.
    """

    def test_in_envelope_returns_same_object(self) -> None:
        pose = _make_pose(0.0, 0.0, 0.0)
        result = clamp_head_pose(pose)
        # When no clamping is needed the function returns the original array
        assert result is pose

    def test_out_of_envelope_returns_new_object(self) -> None:
        _clamped_axes_seen.discard("yaw")
        exceed_deg = math.degrees(HEAD_YAW_MAX_RAD) + 20.0
        pose = _make_pose(yaw=exceed_deg)
        result = clamp_head_pose(pose)
        assert result is not pose  # should be a new array (copy)

    def test_loop_call_cheap(self) -> None:
        """Calling clamp_head_pose in a tight loop shouldn't error."""
        pose = _make_pose(5.0, -10.0, 20.0)
        for _ in range(10_000):
            clamp_head_pose(pose)
