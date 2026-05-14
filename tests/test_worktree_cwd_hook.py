"""Tests for the worktree CWD-leak guard hook (issue #305).

Drives ``.claude/hooks/check_worktree_cwd.py`` as a subprocess with crafted
Claude Code hook payloads on stdin and asserts on its exit code / stderr.
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
from pathlib import Path
from collections.abc import Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK = REPO_ROOT / ".claude" / "hooks" / "check_worktree_cwd.py"


def _run(payload: Mapping[str, object], repo_root: Path, bypass: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("CLAUDE_HOOK_BYPASS", None)
    env["CLAUDE_HOOK_REPO_ROOT"] = str(repo_root)
    if bypass:
        env["CLAUDE_HOOK_BYPASS"] = "1"
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / ".claude" / "worktrees").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def fake_repo_with_worktree(fake_repo: Path) -> Path:
    wt = fake_repo / ".claude" / "worktrees" / "agent-abc123"
    wt.mkdir()
    # Real git worktree dirs have a `.git` file pointing at the gitdir;
    # the hook treats dirs without it as stale leftovers.
    (wt / ".git").write_text("gitdir: ../../../.git/worktrees/agent-abc123\n")
    return fake_repo


def test_allows_when_no_worktrees(fake_repo: Path) -> None:
    payload = {
        "cwd": str(fake_repo),
        "tool_input": {"command": "git checkout -b fix/123-x"},
    }
    result = _run(payload, fake_repo)
    assert result.returncode == 0, result.stderr


def test_blocks_checkout_b_from_main_with_active_worktree(
    fake_repo_with_worktree: Path,
) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git fetch origin main && git checkout -b fix/305-x"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 2
    assert "worktree CWD leak guard" in result.stderr
    assert "#305" in result.stderr
    assert "agent-abc123" in result.stderr


def test_blocks_switch_c_from_main_with_active_worktree(
    fake_repo_with_worktree: Path,
) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git switch -c feat/new-thing"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 2


def test_blocks_worktree_add_from_main_with_active_worktree(
    fake_repo_with_worktree: Path,
) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git worktree add ../other feature/x"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 2


def test_allows_from_worktree_cwd(fake_repo_with_worktree: Path) -> None:
    worktree_path = fake_repo_with_worktree / ".claude" / "worktrees" / "agent-abc123"
    payload = {
        "cwd": str(worktree_path),
        "tool_input": {"command": "git checkout -b fix/305-x"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0, result.stderr


def test_allows_non_bootstrap_command_from_main(fake_repo_with_worktree: Path) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git status"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0, result.stderr


def test_allows_existing_branch_checkout_from_main(
    fake_repo_with_worktree: Path,
) -> None:
    # `git checkout main` is operator behavior, not the bootstrap leak pattern.
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git checkout main"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0, result.stderr


def test_bypass_env_allows_otherwise_blocked_command(
    fake_repo_with_worktree: Path,
) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git checkout -b fix/305-x"},
    }
    result = _run(payload, fake_repo_with_worktree, bypass=True)
    assert result.returncode == 0, result.stderr


def test_empty_stdin_is_allowed(fake_repo_with_worktree: Path) -> None:
    env = os.environ.copy()
    env.pop("CLAUDE_HOOK_BYPASS", None)
    env["CLAUDE_HOOK_REPO_ROOT"] = str(fake_repo_with_worktree)
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="",
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0


def test_malformed_json_is_allowed(fake_repo_with_worktree: Path) -> None:
    env = os.environ.copy()
    env.pop("CLAUDE_HOOK_BYPASS", None)
    env["CLAUDE_HOOK_REPO_ROOT"] = str(fake_repo_with_worktree)
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not valid json{{",
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0


def test_missing_cwd_is_allowed(fake_repo_with_worktree: Path) -> None:
    payload = {"tool_input": {"command": "git checkout -b fix/305-x"}}
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0


def test_missing_command_is_allowed(fake_repo_with_worktree: Path) -> None:
    payload = {"cwd": str(fake_repo_with_worktree), "tool_input": {}}
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0


def test_stale_empty_worktree_dir_is_ignored(fake_repo: Path) -> None:
    """An empty agent-* dir (no .git marker) should not count as active."""
    (fake_repo / ".claude" / "worktrees" / "agent-stale").mkdir()
    payload = {
        "cwd": str(fake_repo),
        "tool_input": {"command": "git checkout -b fix/305-x"},
    }
    result = _run(payload, fake_repo)
    assert result.returncode == 0, result.stderr


def test_pattern_inside_heredoc_is_allowed(fake_repo_with_worktree: Path) -> None:
    """A commit message body that quotes `git checkout -b` is not the leak."""
    body = (
        "git commit -m \"$(cat <<'EOF'\n"
        "fix: explain the pattern\n"
        "\n"
        "Blocks `git checkout -b NEW` from the main checkout.\n"
        "EOF\n"
        ')"'
    )
    payload = {"cwd": str(fake_repo_with_worktree), "tool_input": {"command": body}}
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0, result.stderr


def test_pattern_inside_double_quotes_is_allowed(
    fake_repo_with_worktree: Path,
) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": 'echo "running git checkout -b foo now"'},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0, result.stderr


def test_pattern_inside_single_quotes_is_allowed(
    fake_repo_with_worktree: Path,
) -> None:
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "echo 'git checkout -b inside-string'"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 0, result.stderr


def test_chained_bootstrap_command_is_blocked(
    fake_repo_with_worktree: Path,
) -> None:
    """The canonical agent-bootstrap chain must still trigger the block."""
    payload = {
        "cwd": str(fake_repo_with_worktree),
        "tool_input": {"command": "git fetch origin main && git checkout -b fix/305-x"},
    }
    result = _run(payload, fake_repo_with_worktree)
    assert result.returncode == 2, result.stderr


def test_production_hook_path_exists() -> None:
    """Sanity: the hook is wired in the real settings.json."""
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
    pre = settings["hooks"]["PreToolUse"]
    assert any(
        h.get("matcher") == "Bash"
        and any("check_worktree_cwd.py" in hook.get("command", "") for hook in h.get("hooks", []))
        for h in pre
    )
