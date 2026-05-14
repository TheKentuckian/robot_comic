"""Tests for journald log retention config and install script.

Checks that:
  - deploy/systemd/journald-robot-comic.conf exists and contains all required keys.
  - scripts/install-pi-journald.sh exists.
  - The install script has the executable bit set in the git index (works on Windows).

POSIX-only assertions (os.access execute check) are skipped on Windows.
"""

from __future__ import annotations
import os
import sys
import stat
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DROP_IN = REPO_ROOT / "deploy" / "systemd" / "journald-robot-comic.conf"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install-pi-journald.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_mode(path: Path) -> int:
    """Return the git index mode for *path* (octal int, e.g. 0o100755).

    Works on Windows where the filesystem does not expose POSIX execute bits.
    Returns 0 if the file is not tracked by git.
    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "-s", rel],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    line = result.stdout.strip()
    if not line:
        return 0
    # Output format: "<mode> <hash> <stage>\t<path>"
    return int(line.split()[0], 8)


# ---------------------------------------------------------------------------
# Drop-in config — existence
# ---------------------------------------------------------------------------


def test_drop_in_config_exists() -> None:
    """deploy/systemd/journald-robot-comic.conf must be present in the repo."""
    assert DROP_IN.is_file(), f"Drop-in config not found: {DROP_IN}"


# ---------------------------------------------------------------------------
# Drop-in config — required keys
# ---------------------------------------------------------------------------


def _read_drop_in() -> str:
    return DROP_IN.read_text(encoding="utf-8")


def test_drop_in_contains_system_max_use() -> None:
    assert "SystemMaxUse" in _read_drop_in(), "journald drop-in must set SystemMaxUse"


def test_drop_in_contains_system_keep_free() -> None:
    assert "SystemKeepFree" in _read_drop_in(), "journald drop-in must set SystemKeepFree"


def test_drop_in_contains_system_max_file_size() -> None:
    assert "SystemMaxFileSize" in _read_drop_in(), "journald drop-in must set SystemMaxFileSize"


def test_drop_in_contains_max_retention_sec() -> None:
    assert "MaxRetentionSec" in _read_drop_in(), "journald drop-in must set MaxRetentionSec"


def test_drop_in_has_journal_section_header() -> None:
    """The drop-in must be a valid journald config fragment with [Journal] header."""
    assert "[Journal]" in _read_drop_in(), "journald drop-in must contain [Journal] section header"


# ---------------------------------------------------------------------------
# Install script — existence
# ---------------------------------------------------------------------------


def test_install_script_exists() -> None:
    """scripts/install-pi-journald.sh must be present in the repo."""
    assert INSTALL_SCRIPT.is_file(), f"Install script not found: {INSTALL_SCRIPT}"


# ---------------------------------------------------------------------------
# Install script — executable bit
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="os.access X_OK unreliable for shell scripts on Windows")
def test_install_script_is_executable_posix() -> None:
    """The install script must have the execute bit set (POSIX filesystem check)."""
    assert os.access(INSTALL_SCRIPT, os.X_OK), f"Install script is not executable (chmod +x): {INSTALL_SCRIPT}"


def test_install_script_is_executable_git_index() -> None:
    """The install script must have mode 100755 in the git index (cross-platform)."""
    mode = _git_mode(INSTALL_SCRIPT)
    assert mode != 0, f"Install script is not tracked by git: {INSTALL_SCRIPT}"
    assert mode & stat.S_IXUSR, (
        f"Install script is not marked +x in git index "
        f"(mode={oct(mode)}). Run: git update-index --chmod=+x scripts/install-pi-journald.sh"
    )


# ---------------------------------------------------------------------------
# Install script — content sanity checks
# ---------------------------------------------------------------------------


def _read_script() -> str:
    return INSTALL_SCRIPT.read_text(encoding="utf-8")


def test_install_script_has_shebang() -> None:
    content = _read_script()
    assert content.startswith("#!/"), "Install script must start with a shebang line"


def test_install_script_checks_root() -> None:
    """Script must guard against being run without root privileges."""
    content = _read_script()
    assert "EUID" in content, "Install script must check $EUID (root privilege guard)"


def test_install_script_restarts_journald() -> None:
    content = _read_script()
    assert "systemctl restart systemd-journald" in content, (
        "Install script must restart systemd-journald to apply settings"
    )


def test_install_script_shows_disk_usage() -> None:
    content = _read_script()
    assert "journalctl --disk-usage" in content, "Install script must run 'journalctl --disk-usage' after restart"
