"""Tests for #308 tasks 3 and 4: tracker pitch envelope + default-off.

All tests are pure-logic: no robot hardware, no SDK I/O.
"""

from __future__ import annotations
import sys
import math
import types
import argparse

import pytest

from robot_comic.utils import (
    HEAD_TRACKER_ENV,
    get_requested_head_tracker,
)


# ---------------------------------------------------------------------------
# Task 4: default-off
# ---------------------------------------------------------------------------


class TestHeadTrackerDefaultOff:
    """get_requested_head_tracker must return None when no env var or CLI flag is set."""

    def test_no_env_no_cli_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With REACHY_MINI_HEAD_TRACKER unset and no CLI flag, tracker is off by default."""
        monkeypatch.delenv(HEAD_TRACKER_ENV, raising=False)
        args = argparse.Namespace(head_tracker=None)
        assert get_requested_head_tracker(args) is None

    def test_env_off_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit 'off' value also disables the tracker."""
        monkeypatch.setenv(HEAD_TRACKER_ENV, "off")
        args = argparse.Namespace(head_tracker=None)
        assert get_requested_head_tracker(args) is None

    def test_env_mediapipe_enables_tracker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting REACHY_MINI_HEAD_TRACKER=mediapipe opts back in to tracking."""
        monkeypatch.setenv(HEAD_TRACKER_ENV, "mediapipe")
        args = argparse.Namespace(head_tracker=None)
        assert get_requested_head_tracker(args) == "mediapipe"

    def test_env_yolo_enables_tracker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting REACHY_MINI_HEAD_TRACKER=yolo opts in to the YOLO backend."""
        monkeypatch.setenv(HEAD_TRACKER_ENV, "yolo")
        args = argparse.Namespace(head_tracker=None)
        assert get_requested_head_tracker(args) == "yolo"

    def test_cli_mediapipe_overrides_unset_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit CLI --head-tracker mediapipe enables tracking even when env is absent."""
        monkeypatch.delenv(HEAD_TRACKER_ENV, raising=False)
        args = argparse.Namespace(head_tracker="mediapipe")
        assert get_requested_head_tracker(args) == "mediapipe"

    def test_cli_overrides_env_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI flag takes priority over an env-based disable."""
        monkeypatch.setenv(HEAD_TRACKER_ENV, "off")
        args = argparse.Namespace(head_tracker="mediapipe")
        assert get_requested_head_tracker(args) == "mediapipe"


# ---------------------------------------------------------------------------
# Task 3: pitch envelope env-knob parsing
# ---------------------------------------------------------------------------


def _reload_motion_safety(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> types.ModuleType:
    """Re-import motion_safety with a fresh environment (for module-level constant tests)."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Remove cached module so constants are re-evaluated from env
    monkeypatch.delitem(sys.modules, "robot_comic.motion_safety", raising=False)
    import robot_comic.motion_safety as ms

    return ms


class TestTrackerPitchEnvKnobs:
    """REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD / MAX_RAD are parsed correctly."""

    def test_defaults_are_conservative(self) -> None:
        """Default pitch envelope is ±0.3 rad (~±17°), tighter than the global envelope."""
        import robot_comic.motion_safety as ms

        # Conservative defaults: tighter than the global ±0.524/+0.436 rad limits
        assert ms.HEAD_TRACKER_PITCH_MIN_RAD >= -0.4  # no looser than −23°
        assert ms.HEAD_TRACKER_PITCH_MAX_RAD <= 0.4  # no looser than +23°

    def test_radian_min_knob_is_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD overrides the pitch minimum."""
        ms = _reload_motion_safety(monkeypatch, {"REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD": "-0.2"})
        assert abs(ms.HEAD_TRACKER_PITCH_MIN_RAD - (-0.2)) < 1e-6

    def test_radian_max_knob_is_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REACHY_MINI_HEAD_TRACK_PITCH_MAX_RAD overrides the pitch maximum."""
        ms = _reload_motion_safety(monkeypatch, {"REACHY_MINI_HEAD_TRACK_PITCH_MAX_RAD": "0.25"})
        assert abs(ms.HEAD_TRACKER_PITCH_MAX_RAD - 0.25) < 1e-6

    def test_radian_min_knob_takes_precedence_over_degree_knob(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both radian and degree knobs are set, the radian knob wins for pitch min."""
        ms = _reload_motion_safety(
            monkeypatch,
            {
                "REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD": "-0.15",
                "REACHY_MINI_HEAD_TRACKER_PITCH_MIN_DEG": "-25",
            },
        )
        assert abs(ms.HEAD_TRACKER_PITCH_MIN_RAD - (-0.15)) < 1e-6

    def test_radian_max_knob_takes_precedence_over_degree_knob(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both radian and degree knobs are set, the radian knob wins for pitch max."""
        ms = _reload_motion_safety(
            monkeypatch,
            {
                "REACHY_MINI_HEAD_TRACK_PITCH_MAX_RAD": "0.18",
                "REACHY_MINI_HEAD_TRACKER_PITCH_MAX_DEG": "25",
            },
        )
        assert abs(ms.HEAD_TRACKER_PITCH_MAX_RAD - 0.18) < 1e-6

    def test_invalid_radian_knob_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unparseable radian knob logs a warning and falls back to safe default."""
        import logging

        with caplog.at_level(logging.WARNING, logger="robot_comic.motion_safety"):
            ms = _reload_motion_safety(monkeypatch, {"REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD": "not_a_number"})

        assert ms.HEAD_TRACKER_PITCH_MIN_RAD == pytest.approx(-0.3, abs=1e-6)
        assert any("invalid value" in r.message.lower() for r in caplog.records)

    def test_degree_knob_used_when_radian_knob_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the radian knob is absent, the degree knob is honoured."""
        monkeypatch.delenv("REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD", raising=False)
        ms = _reload_motion_safety(monkeypatch, {"REACHY_MINI_HEAD_TRACKER_PITCH_MIN_DEG": "-20"})
        expected = math.radians(-20.0)
        assert abs(ms.HEAD_TRACKER_PITCH_MIN_RAD - expected) < 1e-6


# ---------------------------------------------------------------------------
# Task 3: pitch clamp engages
# ---------------------------------------------------------------------------


class TestTrackerPitchClampEngages:
    """clamp_tracker_rotation_offsets must clamp pitch when it exceeds the envelope."""

    def test_pitch_above_max_is_clamped(self) -> None:
        """A tracker pitch offset above HEAD_TRACKER_PITCH_MAX_RAD is clamped to the boundary."""
        from robot_comic.motion_safety import (
            HEAD_TRACKER_PITCH_MAX_RAD,
            get_and_reset_clamp_stats,
            clamp_tracker_rotation_offsets,  # noqa: PLC0415
        )

        get_and_reset_clamp_stats()
        overshoot = HEAD_TRACKER_PITCH_MAX_RAD + 0.5
        _, pitch_out, _ = clamp_tracker_rotation_offsets((0.0, overshoot, 0.0))
        assert pitch_out == pytest.approx(HEAD_TRACKER_PITCH_MAX_RAD)

    def test_pitch_below_min_is_clamped(self) -> None:
        """A tracker pitch offset below HEAD_TRACKER_PITCH_MIN_RAD is clamped to the boundary."""
        from robot_comic.motion_safety import (
            HEAD_TRACKER_PITCH_MIN_RAD,
            get_and_reset_clamp_stats,
            clamp_tracker_rotation_offsets,
        )

        get_and_reset_clamp_stats()
        undershoot = HEAD_TRACKER_PITCH_MIN_RAD - 0.5
        _, pitch_out, _ = clamp_tracker_rotation_offsets((0.0, undershoot, 0.0))
        assert pitch_out == pytest.approx(HEAD_TRACKER_PITCH_MIN_RAD)

    def test_pitch_within_envelope_unchanged(self) -> None:
        """A pitch offset within the envelope passes through without modification."""
        from robot_comic.motion_safety import (
            HEAD_TRACKER_PITCH_MAX_RAD,
            HEAD_TRACKER_PITCH_MIN_RAD,
            get_and_reset_clamp_stats,
            clamp_tracker_rotation_offsets,
        )

        get_and_reset_clamp_stats()
        mid = (HEAD_TRACKER_PITCH_MIN_RAD + HEAD_TRACKER_PITCH_MAX_RAD) / 2
        _, pitch_out, _ = clamp_tracker_rotation_offsets((0.0, mid, 0.0))
        assert pitch_out == pytest.approx(mid)

    def test_pitch_clamp_increments_tracker_clamp_counter(self) -> None:
        """Pitch clamping increments the tracker_clamps counter in ClampStats."""
        from robot_comic.motion_safety import (
            HEAD_TRACKER_PITCH_MAX_RAD,
            get_and_reset_clamp_stats,
            clamp_tracker_rotation_offsets,
        )

        get_and_reset_clamp_stats()
        clamp_tracker_rotation_offsets((0.0, HEAD_TRACKER_PITCH_MAX_RAD + 1.0, 0.0))
        stats = get_and_reset_clamp_stats()
        assert stats.tracker_clamps >= 1

    def test_pitch_clamp_does_not_disturb_roll_or_yaw(self) -> None:
        """Clamping pitch must not modify roll or yaw offsets."""
        from robot_comic.motion_safety import (
            HEAD_TRACKER_PITCH_MAX_RAD,
            clamp_tracker_rotation_offsets,
        )

        roll_in, yaw_in = 0.1, -0.2
        roll_out, _, yaw_out = clamp_tracker_rotation_offsets((roll_in, HEAD_TRACKER_PITCH_MAX_RAD + 0.5, yaw_in))
        assert roll_out == pytest.approx(roll_in)
        assert yaw_out == pytest.approx(yaw_in)

    def test_pitch_clamp_tight_envelope_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Operator can tighten the pitch envelope below the default via env knob."""
        ms = _reload_motion_safety(
            monkeypatch,
            {
                "REACHY_MINI_HEAD_TRACK_PITCH_MIN_RAD": "-0.1",
                "REACHY_MINI_HEAD_TRACK_PITCH_MAX_RAD": "0.1",
            },
        )
        # With a ±0.1 rad envelope, 0.2 rad should be clamped to 0.1 rad
        _, pitch_out, _ = ms.clamp_tracker_rotation_offsets((0.0, 0.2, 0.0))
        assert pitch_out == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Task 3: startup log emitted
# ---------------------------------------------------------------------------


class TestStartupPitchEnvelopeLog:
    """log_tracker_pitch_envelope emits an INFO line with the resolved limits."""

    def test_log_emitted_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_tracker_pitch_envelope should emit at INFO level with pitch values."""
        import logging

        from robot_comic.motion_safety import (
            HEAD_TRACKER_PITCH_MAX_RAD,
            HEAD_TRACKER_PITCH_MIN_RAD,
            log_tracker_pitch_envelope,
        )

        with caplog.at_level(logging.INFO, logger="robot_comic.motion_safety"):
            log_tracker_pitch_envelope()

        assert any("tracker pitch envelope" in r.message.lower() for r in caplog.records)
        # The log must contain the resolved values (as text substrings)
        matching = [r for r in caplog.records if "tracker pitch envelope" in r.message.lower()]
        assert matching, "No matching log record found"
        log_text = matching[0].message
        assert f"{HEAD_TRACKER_PITCH_MIN_RAD:.4f}" in log_text
        assert f"{HEAD_TRACKER_PITCH_MAX_RAD:.4f}" in log_text
