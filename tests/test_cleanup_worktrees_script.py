"""Tests for scripts/cleanup-worktrees.sh.

The script is a POSIX bash script and cannot be exercised on Windows.
Tests that invoke the script are skipped on win32 — CI runs them on the
Linux matrix leg.

The file-exists and executable checks run on all platforms where the check
is meaningful; on Windows ``os.access(..., os.X_OK)`` is not reliable for
shell scripts so we skip that assertion on win32 as well.
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cleanup-worktrees.sh"

_TIMEOUT = 15  # seconds — generous for slow CI


# ---------------------------------------------------------------------------
# Platform-agnostic: script file exists
# ---------------------------------------------------------------------------


def test_script_file_exists() -> None:
    """The cleanup script must be present in scripts/."""
    assert SCRIPT.exists(), f"Script not found: {SCRIPT}"


@pytest.mark.skipif(sys.platform == "win32", reason="os.X_OK unreliable on Windows for shell scripts")
def test_script_is_executable() -> None:
    """The cleanup script must have the executable bit set (POSIX)."""
    assert os.access(SCRIPT, os.X_OK), f"Script is not executable: {SCRIPT}"


# ---------------------------------------------------------------------------
# Smoke tests — require bash (skipped on Windows)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_help_exits_zero() -> None:
    """--help prints usage text and exits 0."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}.\nstderr: {result.stderr}"
    # Usage output should mention the script name or key options
    combined = result.stdout + result.stderr
    assert "usage" in combined.lower() or "--all" in combined.lower() or "cleanup" in combined.lower(), (
        f"Expected usage text in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_dry_run_exits_zero_in_non_agent_repo(tmp_path: Path) -> None:
    """--dry-run in a repo with no agent worktrees exits 0 and reports nothing to remove."""
    # Initialise a bare git repo so the script can call `git worktree list`
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "t@t.com",
        },
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    assert result.returncode == 0, (
        f"Expected exit 0, got {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    # Should say nothing to do / found 0 worktrees
    assert "nothing to do" in combined.lower() or "0" in combined, (
        f"Expected 'nothing to do' or zero count in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_unknown_flag_exits_nonzero() -> None:
    """An unrecognised flag should produce a non-zero exit code."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "--bogus-flag-that-does-not-exist"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    assert result.returncode != 0, f"Expected non-zero exit for unknown flag, got 0.\nstdout: {result.stdout}"
    combined = result.stdout + result.stderr
    assert "unknown" in combined.lower() or "error" in combined.lower() or "bogus" in combined.lower(), (
        f"Expected error message for unknown flag.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
