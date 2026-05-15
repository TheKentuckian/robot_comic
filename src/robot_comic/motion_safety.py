"""Head-motion safety helpers for Reachy Mini.

This module provides two layers of protection against hardware impacts:

1. **Per-axis soft clamp** — `clamp_head_pose()` extracts the RPY angles from
   a 4x4 SE3 head-pose matrix, clamps each axis to a configurable safe
   envelope, and rebuilds the matrix.  The clamp is applied once per control-
   loop tick inside `MovementManager._issue_control_command`, so it catches
   every move source (tool calls, dances, breathing, goto).

2. **Angular-velocity cap** — `cap_head_velocity()` limits the per-tick
   angular step (in radians per loop tick) when advancing toward a target
   pose.  This prevents step-changes arriving at joints at max servo velocity,
   which can cause the head to slam into the plastic cowling.

Safe-envelope defaults
----------------------
Values are derived from:

* ``hardware_config.yaml`` Stewart-platform motor limits (s1/s3 ≈ −48 to +80 deg;
  s2/s4/s6 ≈ −80 to +(48–70) deg in individual-motor frame).
* IK module constant ``max_relative_yaw = deg2rad(65)`` (world frame).
* Observed safe motion range from MoveHead tool (yaw ±40 deg, pitch ±30 deg).
* A 5-degree inward margin on each observed limit to provide a cowling
  clearance buffer.

Resulting soft limits (world frame, radians):

  pitch : [−0.524, +0.436] (−30 deg up / +25 deg down)
  yaw   : [−0.785, +0.785] (±45 deg)
  roll  : [−0.349, +0.349] (±20 deg)

All limits are also configurable via environment variables so they can be
tightened or relaxed without a code change:

  REACHY_MINI_HEAD_PITCH_MIN_DEG   (default −30)
  REACHY_MINI_HEAD_PITCH_MAX_DEG   (default  25)
  REACHY_MINI_HEAD_YAW_MIN_DEG     (default −45)
  REACHY_MINI_HEAD_YAW_MAX_DEG     (default  45)
  REACHY_MINI_HEAD_ROLL_MIN_DEG    (default −20)
  REACHY_MINI_HEAD_ROLL_MAX_DEG    (default  20)
  REACHY_MINI_HEAD_MAX_VEL_RAD_S   (default 1.5; back-compat fallback)

Per-axis velocity caps (#272) override the single ``HEAD_MAX_VEL_RAD_S`` knob
when set, so operators can tune each axis independently:

  REACHY_MINI_HEAD_PITCH_MAX_VEL_RAD_S   (default = HEAD_MAX_VEL_RAD_S)
  REACHY_MINI_HEAD_YAW_MAX_VEL_RAD_S     (default = HEAD_MAX_VEL_RAD_S)
  REACHY_MINI_HEAD_ROLL_MAX_VEL_RAD_S    (default = HEAD_MAX_VEL_RAD_S)
"""

from __future__ import annotations
import os
import math
import logging
from typing import TYPE_CHECKING, Set, Dict
from dataclasses import dataclass


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-variable helpers
# ---------------------------------------------------------------------------


def _env_deg(name: str, default_deg: float) -> float:
    """Read an angle env var in degrees; return the value in radians."""
    raw = os.getenv(name)
    if raw is None:
        return math.radians(default_deg)
    try:
        return math.radians(float(raw.strip()))
    except ValueError:
        logger.warning(
            "motion_safety: invalid value %r for %s, using default %.1f deg",
            raw,
            name,
            default_deg,
        )
        return math.radians(default_deg)


def _env_float(name: str, default: float) -> float:
    """Read a float env var."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning(
            "motion_safety: invalid value %r for %s, using default %.3f",
            raw,
            name,
            default,
        )
        return default


# ---------------------------------------------------------------------------
# Safe-envelope constants (read once at import time; can be overridden via env)
# ---------------------------------------------------------------------------

HEAD_PITCH_MIN_RAD: float = _env_deg("REACHY_MINI_HEAD_PITCH_MIN_DEG", -30.0)  # looking up
HEAD_PITCH_MAX_RAD: float = _env_deg("REACHY_MINI_HEAD_PITCH_MAX_DEG", 25.0)  # looking down
HEAD_YAW_MIN_RAD: float = _env_deg("REACHY_MINI_HEAD_YAW_MIN_DEG", -45.0)  # right
HEAD_YAW_MAX_RAD: float = _env_deg("REACHY_MINI_HEAD_YAW_MAX_DEG", 45.0)  # left
HEAD_ROLL_MIN_RAD: float = _env_deg("REACHY_MINI_HEAD_ROLL_MIN_DEG", -20.0)
HEAD_ROLL_MAX_RAD: float = _env_deg("REACHY_MINI_HEAD_ROLL_MAX_DEG", 20.0)

# Maximum angular velocity for any single RPY axis (rad/s) — back-compat knob
# used as the per-axis default. At 60 Hz, 1.5 rad/s → ≈0.025 rad per tick ≈ 1.4
# deg per tick.
HEAD_MAX_VEL_RAD_S: float = _env_float("REACHY_MINI_HEAD_MAX_VEL_RAD_S", 1.5)

# Per-axis caps (#272): operators can tune pitch/yaw/roll independently.
# Defaults fall through to HEAD_MAX_VEL_RAD_S so behaviour is unchanged unless
# an axis-specific override is set.
HEAD_PITCH_MAX_VEL_RAD_S: float = _env_float("REACHY_MINI_HEAD_PITCH_MAX_VEL_RAD_S", HEAD_MAX_VEL_RAD_S)
HEAD_YAW_MAX_VEL_RAD_S: float = _env_float("REACHY_MINI_HEAD_YAW_MAX_VEL_RAD_S", HEAD_MAX_VEL_RAD_S)
HEAD_ROLL_MAX_VEL_RAD_S: float = _env_float("REACHY_MINI_HEAD_ROLL_MAX_VEL_RAD_S", HEAD_MAX_VEL_RAD_S)

# Continuous-tracker safe envelope (#308 hypothesis 3): the head tracker can
# push the composed pose to the edge of the global envelope on every frame;
# composition with breathing / wobbler can then tip over into IK-invalid
# territory and the daemon logs "Collision detected or head pose not
# achievable!". Constraining the tracker OFFSET before composition cuts those
# warnings off at the source. Defaults match the global envelope so this is a
# no-op until operators tighten per-unit via env.
HEAD_TRACKER_PITCH_MIN_RAD: float = _env_deg("REACHY_MINI_HEAD_TRACKER_PITCH_MIN_DEG", -30.0)
HEAD_TRACKER_PITCH_MAX_RAD: float = _env_deg("REACHY_MINI_HEAD_TRACKER_PITCH_MAX_DEG", 25.0)
HEAD_TRACKER_YAW_MIN_RAD: float = _env_deg("REACHY_MINI_HEAD_TRACKER_YAW_MIN_DEG", -45.0)
HEAD_TRACKER_YAW_MAX_RAD: float = _env_deg("REACHY_MINI_HEAD_TRACKER_YAW_MAX_DEG", 45.0)
HEAD_TRACKER_ROLL_MIN_RAD: float = _env_deg("REACHY_MINI_HEAD_TRACKER_ROLL_MIN_DEG", -20.0)
HEAD_TRACKER_ROLL_MAX_RAD: float = _env_deg("REACHY_MINI_HEAD_TRACKER_ROLL_MAX_DEG", 20.0)

# Track which axes have been clamped so we only emit one DEBUG per session.
_clamped_axes_seen: Set[str] = set()


# ---------------------------------------------------------------------------
# Clamp-event counters (#272)
# ---------------------------------------------------------------------------


@dataclass
class ClampStats:
    """Running counts of pose-clamp, velocity-cap, and tracker-clamp events."""

    pose_clamps: Dict[str, int]  # axis name → count of clamps applied this window
    velocity_caps: int  # total velocity-cap engagements this window
    tracker_clamps: int = 0  # tracker safe-envelope engagements this window (#308)

    def is_empty(self) -> bool:
        """Return True when no clamps or caps fired in the window."""
        return self.velocity_caps == 0 and self.tracker_clamps == 0 and not any(self.pose_clamps.values())


_pose_clamp_counts: Dict[str, int] = {"roll": 0, "pitch": 0, "yaw": 0}
_velocity_cap_count: int = 0
_tracker_clamp_count: int = 0


def get_and_reset_clamp_stats() -> ClampStats:
    """Return clamp counters since last call, then zero them.

    The MovementManager tick loop calls this on a periodic schedule to emit a
    summary log line; tests use it to assert clamps fired.
    """
    global _velocity_cap_count, _tracker_clamp_count
    snapshot = ClampStats(
        pose_clamps=dict(_pose_clamp_counts),
        velocity_caps=_velocity_cap_count,
        tracker_clamps=_tracker_clamp_count,
    )
    for key in _pose_clamp_counts:
        _pose_clamp_counts[key] = 0
    _velocity_cap_count = 0
    _tracker_clamp_count = 0
    return snapshot


def clamp_tracker_rotation_offsets(
    rotation_xyz_rad: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Clamp tracker rotation offsets (roll, pitch, yaw) to the tracker envelope (#308).

    Applied to the camera-worker face-tracking offset before it joins the
    secondary-move composition, so the daemon never sees an out-of-envelope
    target from the continuous tracker. Increments ``_tracker_clamp_count``
    when any axis is clamped so the periodic summary surfaces the activity.
    """
    global _tracker_clamp_count
    roll, pitch, yaw = rotation_xyz_rad
    new_roll = max(HEAD_TRACKER_ROLL_MIN_RAD, min(HEAD_TRACKER_ROLL_MAX_RAD, roll))
    new_pitch = max(HEAD_TRACKER_PITCH_MIN_RAD, min(HEAD_TRACKER_PITCH_MAX_RAD, pitch))
    new_yaw = max(HEAD_TRACKER_YAW_MIN_RAD, min(HEAD_TRACKER_YAW_MAX_RAD, yaw))
    if new_roll != roll or new_pitch != pitch or new_yaw != yaw:
        _tracker_clamp_count += 1
    return (new_roll, new_pitch, new_yaw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clamp_head_pose(
    pose: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return a copy of *pose* with roll/pitch/yaw clamped to the safe envelope.

    The function never raises; on any error the original pose is returned
    unchanged so the control loop keeps running.

    Parameters
    ----------
    pose:
        4×4 homogeneous SE3 matrix representing the head pose (world frame).

    Returns
    -------
    NDArray
        Clamped 4×4 pose matrix of the same dtype as the input.

    """
    from scipy.spatial.transform import Rotation

    try:
        r = Rotation.from_matrix(pose[:3, :3])
        # Extrinsic XYZ = intrinsic roll, pitch, yaw (matches create_head_pose convention)
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)

        clamped_roll = _clamp_axis("roll", roll, HEAD_ROLL_MIN_RAD, HEAD_ROLL_MAX_RAD)
        clamped_pitch = _clamp_axis("pitch", pitch, HEAD_PITCH_MIN_RAD, HEAD_PITCH_MAX_RAD)
        clamped_yaw = _clamp_axis("yaw", yaw, HEAD_YAW_MIN_RAD, HEAD_YAW_MAX_RAD)

        if clamped_roll == roll and clamped_pitch == pitch and clamped_yaw == yaw:
            return pose  # fast path — no copy needed

        clamped_rot = Rotation.from_euler("xyz", [clamped_roll, clamped_pitch, clamped_yaw], degrees=False)
        result = pose.copy()
        result[:3, :3] = clamped_rot.as_matrix()
        return result

    except Exception:
        logger.debug("clamp_head_pose: unexpected error; returning pose unchanged", exc_info=True)
        return pose


def cap_head_velocity(
    current_pose: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    dt: float,
    max_vel_rad_s: float = HEAD_MAX_VEL_RAD_S,
    *,
    max_vel_roll_rad_s: float | None = None,
    max_vel_pitch_rad_s: float | None = None,
    max_vel_yaw_rad_s: float | None = None,
) -> NDArray[np.floating]:
    """Advance *current_pose* toward *target_pose* by at most max-vel × dt per axis.

    This prevents step-changes in commanded pose from arriving at the servo at
    maximum velocity, which is the root cause of the cowling impact.

    Parameters
    ----------
    current_pose:
        The last-commanded 4×4 head pose.
    target_pose:
        The desired 4×4 head pose for this tick.
    dt:
        Elapsed time since the previous tick (seconds).
    max_vel_rad_s:
        Default per-axis cap (rad/s) applied to any axis without an override.
    max_vel_roll_rad_s:
        Optional roll-axis override (rad/s). When None, ``max_vel_rad_s`` is used.
    max_vel_pitch_rad_s:
        Optional pitch-axis override (rad/s). When None, ``max_vel_rad_s`` is used.
    max_vel_yaw_rad_s:
        Optional yaw-axis override (rad/s). When None, ``max_vel_rad_s`` is used.

    Returns
    -------
    NDArray
        A 4×4 pose matrix advanced no more than its per-axis budget toward
        the target.  Falls back to *target_pose* on error.

    """
    global _velocity_cap_count
    if dt <= 0.0:
        return target_pose

    import numpy as np
    from scipy.spatial.transform import Rotation

    roll_cap = max_vel_roll_rad_s if max_vel_roll_rad_s is not None else max_vel_rad_s
    pitch_cap = max_vel_pitch_rad_s if max_vel_pitch_rad_s is not None else max_vel_rad_s
    yaw_cap = max_vel_yaw_rad_s if max_vel_yaw_rad_s is not None else max_vel_rad_s
    max_steps = (roll_cap * dt, pitch_cap * dt, yaw_cap * dt)

    try:
        r_cur = Rotation.from_matrix(current_pose[:3, :3])
        r_tgt = Rotation.from_matrix(target_pose[:3, :3])

        cur_rpy = r_cur.as_euler("xyz", degrees=False)
        tgt_rpy = r_tgt.as_euler("xyz", degrees=False)

        stepped = np.array(
            [_step_toward(cur, tgt, step) for cur, tgt, step in zip(cur_rpy, tgt_rpy, max_steps)],
            dtype=float,
        )

        if np.allclose(stepped, tgt_rpy, atol=1e-9):
            return target_pose  # fast path — already at target

        _velocity_cap_count += 1

        new_rot = Rotation.from_euler("xyz", stepped, degrees=False)
        result = target_pose.copy()
        result[:3, :3] = new_rot.as_matrix()

        # Interpolate translation linearly with the same scalar blend factor.
        delta_rpy = np.abs(tgt_rpy - cur_rpy)
        max_delta = float(np.max(delta_rpy))
        if max_delta > 0.0:
            actual_step = float(np.max(np.abs(stepped - cur_rpy)))
            blend = min(1.0, actual_step / max_delta) if max_delta > 1e-9 else 1.0
        else:
            blend = 1.0

        cur_t = current_pose[:3, 3]
        tgt_t = target_pose[:3, 3]
        result[:3, 3] = cur_t + (tgt_t - cur_t) * blend
        return result

    except Exception:
        logger.debug("cap_head_velocity: unexpected error; returning target unchanged", exc_info=True)
        return target_pose


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp_axis(name: str, value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi].  Emits one DEBUG log per axis per session.

    Each clamp also increments ``_pose_clamp_counts[name]`` so the periodic
    summary in MovementManager can surface clamp activity (#272).
    """
    if value < lo:
        _maybe_log_clamp(name, value, lo)
        _pose_clamp_counts[name] = _pose_clamp_counts.get(name, 0) + 1
        return lo
    if value > hi:
        _maybe_log_clamp(name, value, hi)
        _pose_clamp_counts[name] = _pose_clamp_counts.get(name, 0) + 1
        return hi
    return value


def _maybe_log_clamp(name: str, original: float, clamped: float) -> None:
    """Log a DEBUG message the first time each axis is clamped."""
    if name not in _clamped_axes_seen:
        _clamped_axes_seen.add(name)
        logger.debug(
            "motion_safety: head %s clamped %.3f→%.3f rad (%.1f→%.1f deg) [further clamps on this axis suppressed]",
            name,
            original,
            clamped,
            math.degrees(original),
            math.degrees(clamped),
        )


def _step_toward(current: float, target: float, max_step: float) -> float:
    """Move *current* toward *target* by at most *max_step*."""
    diff = target - current
    if abs(diff) <= max_step:
        return target
    return current + math.copysign(max_step, diff)
