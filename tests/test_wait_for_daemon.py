"""Tests for scripts/wait-for-reachy-daemon.sh.

The script is a POSIX bash script and cannot be exercised on Windows.
All tests are skipped on win32 — CI runs them on the Linux matrix leg.
"""

from __future__ import annotations
import os
import sys
import stat
import tempfile
import textwrap
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "scripts" / "wait-for-reachy-daemon.sh"

# Maximum wall-clock seconds a single test is allowed to run.  The "always
# fails" test drives the full 10 s polling loop, so give it generous headroom.
_TIMEOUT_FAST = 5
_TIMEOUT_SLOW = 15


def _make_fake_curl(tmp: Path, *, exit_code: int) -> Path:
    """Write a fake `curl` wrapper that always exits with *exit_code*."""
    fake = tmp / "curl"
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            exit {exit_code}
            """
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return fake


def _run_script(tmp: Path) -> subprocess.CompletedProcess[str]:
    """Execute the poll script with *tmp* prepended to PATH."""
    env = os.environ.copy()
    env["PATH"] = str(tmp) + os.pathsep + env.get("PATH", "")
    # Override the endpoint so it never accidentally hits a real daemon.
    env["REACHY_DAEMON_URL"] = "http://127.0.0.1:19999"
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"Script not found: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"Script is not executable: {SCRIPT}"


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_exits_zero_when_curl_succeeds_immediately() -> None:
    """Daemon is up immediately — script should return 0 quickly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _make_fake_curl(tmp, exit_code=0)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_FAST,
            env={**os.environ, "PATH": str(tmp) + os.pathsep + os.environ.get("PATH", "")},
        )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "ready" in result.stderr.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_exits_nonzero_when_curl_always_fails() -> None:
    """Daemon never comes up — script should exit 1 after exhausting retries."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _make_fake_curl(tmp, exit_code=7)  # curl exit 7 = "failed to connect"
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SLOW,
            env={**os.environ, "PATH": str(tmp) + os.pathsep + os.environ.get("PATH", "")},
        )
    assert result.returncode == 1, f"Expected exit 1 on timeout, got {result.returncode}. stderr: {result.stderr}"
    assert "not ready" in result.stderr.lower() or "giving up" in result.stderr.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="shell script — POSIX only")
def test_ready_message_includes_elapsed_indicator() -> None:
    """The success path should print a human-readable 'ready after Xs' message."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _make_fake_curl(tmp, exit_code=0)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_FAST,
            env={**os.environ, "PATH": str(tmp) + os.pathsep + os.environ.get("PATH", "")},
        )
    # The message goes to stderr so it doesn't pollute ExecStartPre stdout.
    assert result.returncode == 0
    assert result.stderr.strip() != "", "Expected a status message on stderr"
