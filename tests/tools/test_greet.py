"""Tests for the Greet tool."""

from __future__ import annotations
import os
import sys
import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


_PROFILE_PATH = Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "greet.py"


def _load_greet_module():
    """Load greet.py from its package path using importlib."""
    spec = importlib.util.spec_from_file_location("don_rickles_greet", _PROFILE_PATH)
    assert spec and spec.loader, f"Cannot load module from {_PROFILE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["don_rickles_greet"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def greet_mod():
    """Module-scoped fixture that loads the greet module from its profile path."""
    return _load_greet_module()


@pytest.fixture
def Greet(greet_mod):
    """Fixture that returns the Greet class from the loaded module."""
    return greet_mod.Greet


@pytest.fixture
def fuzzy_match(greet_mod):
    """Fixture that returns the _fuzzy_match helper from the loaded module."""
    return greet_mod._fuzzy_match


def make_deps() -> MagicMock:
    """Build a minimal mock ToolDependencies with a blank camera frame."""
    deps = MagicMock()
    deps.motion_duration_s = 0.0
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    deps.camera_worker = MagicMock()
    deps.camera_worker.get_latest_frame.return_value = frame
    deps.movement_manager = MagicMock()
    deps.reachy_mini = MagicMock()
    deps.reachy_mini.get_current_head_pose.return_value = MagicMock()
    deps.reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    return deps


# ── scan: face found immediately ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_face_found_immediately(Greet):
    """Face detected on first frame — no sweep triggered."""
    deps = make_deps()
    with patch("don_rickles_greet.MP_AVAILABLE", True), patch("don_rickles_greet._detect_face", return_value=True):
        result = await Greet()(deps, action="scan")
    assert result == {"face_detected": True}
    deps.movement_manager.queue_move.assert_not_called()


# ── scan: face found during sweep ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_face_found_during_sweep(Greet):
    """No face initially; face found on second sweep position (up)."""
    deps = make_deps()
    # Initial frame: no face. Second call (left sweep position): no face. Third call (up): face found.
    with (
        patch("don_rickles_greet.MP_AVAILABLE", True),
        patch("don_rickles_greet._detect_face", side_effect=[False, False, True]),
        patch("asyncio.sleep"),
    ):
        result = await Greet()(deps, action="scan")
    assert result == {"face_detected": True}


# ── scan: no face after full sweep ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_no_face_after_full_sweep(Greet):
    """Full sweep completes with no face found — returns no_subject."""
    deps = make_deps()
    with (
        patch("don_rickles_greet.MP_AVAILABLE", True),
        patch("don_rickles_greet._detect_face", return_value=False),
        patch("asyncio.sleep"),
    ):
        result = await Greet()(deps, action="scan")
    assert result == {"no_subject": True}


# ── scan: MediaPipe unavailable (fail-open) ───────────────────────────────────


@pytest.mark.asyncio
async def test_scan_mediapipe_unavailable(Greet):
    """MediaPipe not installed — fail-open, assumes face present."""
    deps = make_deps()
    with patch("don_rickles_greet.MP_AVAILABLE", False):
        result = await Greet()(deps, action="scan")
    assert result["face_detected"] is True
    assert "note" in result


# ── scan: camera errors ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_camera_worker_none(Greet):
    """camera_worker is None — returns error."""
    deps = make_deps()
    deps.camera_worker = None
    result = await Greet()(deps, action="scan")
    assert "error" in result


@pytest.mark.asyncio
async def test_scan_no_frame_available(Greet):
    """get_latest_frame returns None — returns error."""
    deps = make_deps()
    deps.camera_worker.get_latest_frame.return_value = None
    result = await Greet()(deps, action="scan")
    assert "error" in result


# ── identify: match found ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_match_found(Greet, tmp_path):
    """Exact name match returns returning=True with full profile."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    (session_dir / "session_20260504_120000.json").write_text(
        json.dumps(
            {
                "session_id": "20260504_120000",
                "name": "Tony",
                "job": "engineer",
                "hometown": "Pittsburgh",
                "details": ["nervous laugh"],
                "last_updated": "2026-05-04T12:00:00",
                "roast_targets_used": [],
            }
        )
    )
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result["returning"] is True
    assert result["name"] == "Tony"
    assert result["profile"]["job"] == "engineer"
    assert result["profile"]["hometown"] == "Pittsburgh"
    assert result["profile"]["details"] == ["nervous laugh"]
    assert "last_seen" in result
    assert "load_instruction" in result


# ── identify: fuzzy match ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_fuzzy_match(Greet, tmp_path):
    """'Jon' matches stored 'John' — score 0.857, above 0.75 threshold."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    (session_dir / "session_20260504_120000.json").write_text(
        json.dumps(
            {
                "session_id": "20260504_120000",
                "name": "John",
                "job": "plumber",
                "hometown": "Brooklyn",
                "details": [],
                "last_updated": "2026-05-04T12:00:00",
                "roast_targets_used": [],
            }
        )
    )
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Jon")
    assert result["returning"] is True
    assert result["name"] == "John"


# ── identify: below threshold ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_below_threshold(Greet, tmp_path):
    """Score below 0.75 threshold — returns returning=False."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    (session_dir / "session_20260504_120000.json").write_text(
        json.dumps(
            {
                "session_id": "20260504_120000",
                "name": "Xiomara",
                "job": "dentist",
                "hometown": "Albuquerque",
                "details": [],
                "last_updated": "2026-05-04T12:00:00",
                "roast_targets_used": [],
            }
        )
    )
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result == {"returning": False}


# ── identify: no sessions ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_no_sessions(Greet, tmp_path):
    """Empty session directory — returns returning=False."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result == {"returning": False}


# ── identify: old sessions ignored ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_ignores_old_sessions(Greet, tmp_path):
    """Session file older than 30 days is excluded from matching."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    old_file = session_dir / "session_19990101_120000.json"
    old_file.write_text(
        json.dumps(
            {
                "session_id": "19990101_120000",
                "name": "Tony",
                "job": "engineer",
                "hometown": "Pittsburgh",
                "details": [],
                "last_updated": "1999-01-01T12:00:00",
                "roast_targets_used": [],
            }
        )
    )
    old_time = 946728000  # ~year 2000, well outside 30-day window
    os.utime(old_file, (old_time, old_time))
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result == {"returning": False}


# ── identify: null names skipped ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_skips_null_names(Greet, tmp_path):
    """Sessions with name=null are skipped — returns returning=False."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    (session_dir / "session_20260504_120000.json").write_text(
        json.dumps(
            {
                "session_id": "20260504_120000",
                "name": None,
                "job": "engineer",
                "hometown": "Pittsburgh",
                "details": [],
                "last_updated": "2026-05-04T12:00:00",
                "roast_targets_used": [],
            }
        )
    )
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result == {"returning": False}


# ── identify: corrupt files skipped ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_skips_corrupt_files(Greet, tmp_path):
    """Corrupt JSON session file is skipped without raising — returns returning=False."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    (session_dir / "session_20260504_120000.json").write_text("not valid json {{{")
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result == {"returning": False}


# ── identify: name required ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_name_required(Greet, tmp_path):
    """Missing name parameter for identify action — returns error."""
    session_dir = tmp_path / ".rickles_sessions"
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify")
    assert "error" in result


# ── identify: callbacks present ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identify_returns_callbacks(Greet, tmp_path):
    """Matched session with job + details produces at least one callback hint."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    (session_dir / "session_20260504_120000.json").write_text(
        json.dumps(
            {
                "session_id": "20260504_120000",
                "name": "Tony",
                "job": "engineer",
                "hometown": "Pittsburgh",
                "details": ["nervous laugh"],
                "last_updated": "2026-05-04T12:00:00",
                "roast_targets_used": [],
            }
        )
    )
    result = await Greet(session_dir=session_dir)(make_deps(), action="identify", name="Tony")
    assert result["returning"] is True
    assert len(result["callbacks"]) >= 1
    assert "Tony" in result["callbacks"][0] or "engineer" in result["callbacks"][0]


# ── unknown action ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_action_returns_error(Greet):
    """Unknown action value returns error dict."""
    result = await Greet()(make_deps(), action="explode")
    assert "error" in result


# ── _fuzzy_match unit tests ───────────────────────────────────────────────────


def test_fuzzy_match_exact(fuzzy_match, tmp_path):
    """Exact name match returns score 1.0."""
    p = tmp_path / "s.json"
    p.touch()
    name, path, score = fuzzy_match("Tony", [("Tony", p)])
    assert name == "Tony"
    assert score == 1.0


def test_fuzzy_match_above_threshold(fuzzy_match, tmp_path):
    """'Jon' vs 'John' scores 0.857, above the 0.75 threshold — returns match."""
    p = tmp_path / "s.json"
    p.touch()
    name, path, score = fuzzy_match("Jon", [("John", p)])
    assert name == "John"
    assert score >= 0.75


def test_fuzzy_match_below_threshold_returns_none(fuzzy_match, tmp_path):
    """Score below threshold — returns (None, None, score)."""
    p = tmp_path / "s.json"
    p.touch()
    name, path, score = fuzzy_match("Tony", [("Xiomara", p)])
    assert name is None
    assert path is None
