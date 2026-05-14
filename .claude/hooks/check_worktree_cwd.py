#!/usr/bin/env python3
"""PreToolUse:Bash hook — guard against the worktree CWD leak (issue #305).

Reads Claude Code hook JSON from stdin. Blocks the canonical agent-bootstrap
leak pattern: a `git checkout -b NEW`, `git switch -c NEW`, or `git worktree
add` running from the MAIN repo checkout's CWD while at least one active
agent worktree exists under ``.claude/worktrees/agent-*``.

Hook protocol (Claude Code):
- stdin: JSON with at least ``cwd`` and ``tool_input.command``
- exit 0: allow (silent)
- exit 2: block; stderr is surfaced to the model

Bypass: set ``CLAUDE_HOOK_BYPASS=1`` to skip the check for one invocation.
Override repo root (tests): set ``CLAUDE_HOOK_REPO_ROOT=/some/path``.
"""

from __future__ import annotations
import os
import re
import sys
import json
from pathlib import Path


def _repo_root() -> Path:
    override = os.environ.get("CLAUDE_HOOK_REPO_ROOT")
    if override:
        return Path(override).resolve()
    # .claude/hooks/check_worktree_cwd.py -> repo root is two parents up
    return Path(__file__).resolve().parent.parent.parent


def _active_worktrees(repo_root: Path) -> list[Path]:
    """Return live worktree dirs under ``.claude/worktrees/``.

    An entry counts as live only if it contains a ``.git`` marker (the file
    that ``git worktree add`` writes). Empty leftover dirs from killed agents
    are ignored — they were the cause of a false-positive in the first
    iteration of this hook.
    """
    wt_root = repo_root / ".claude" / "worktrees"
    if not wt_root.exists():
        return []
    out: list[Path] = []
    for p in sorted(wt_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("agent-"):
            continue
        if not (p / ".git").exists():
            continue
        out.append(p)
    return out


_BOOTSTRAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgit\s+checkout\s+-b\s+\S"),
    re.compile(r"\bgit\s+switch\s+-c\s+\S"),
    re.compile(r"\bgit\s+worktree\s+add\b"),
)


def _looks_like_bootstrap(command: str) -> bool:
    """Detect an agent-bootstrap pattern OUTSIDE quoted/heredoc text.

    The first iteration of this hook matched the pattern literally anywhere
    in the command, which false-positived on commit messages and echo
    arguments that happened to contain the words. This version skips matches
    whose prefix opens a heredoc (``<<``) or sits inside unbalanced single
    or double quotes — good enough for the shell idioms agents and the
    operator use without pulling in a real shell parser.
    """
    for pattern in _BOOTSTRAP_PATTERNS:
        for m in pattern.finditer(command):
            prefix = command[: m.start()]
            if "<<" in prefix:
                continue
            if prefix.count("'") % 2 == 1:
                continue
            if prefix.count('"') % 2 == 1:
                continue
            return True
    return False


def main() -> int:
    """Read a Claude Code PreToolUse:Bash payload from stdin and decide allow/block."""
    if os.environ.get("CLAUDE_HOOK_BYPASS") == "1":
        return 0

    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0  # never break the tool call on a malformed payload

    cwd_raw = payload.get("cwd")
    if not cwd_raw:
        return 0
    try:
        cwd = Path(cwd_raw).resolve()
    except (OSError, ValueError):
        return 0

    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") or ""
    if not isinstance(command, str) or not command:
        return 0

    repo_root = _repo_root()
    if cwd != repo_root:
        return 0  # already inside a worktree (or somewhere else) — not the leak case

    if not _looks_like_bootstrap(command):
        return 0

    worktrees = _active_worktrees(repo_root)
    if not worktrees:
        return 0

    snippet = command.strip().replace("\n", " ")[:200]
    msg = (
        "BLOCKED by worktree CWD leak guard (issue #305).\n"
        f"  Command: {snippet}\n"
        f"  CWD:     {cwd}\n"
        f"  Active worktrees: {[str(w) for w in worktrees]}\n"
        "\n"
        "A branch-creating git command is running from the MAIN repo checkout\n"
        "while at least one agent worktree exists. This is the bootstrap-leak\n"
        "pattern from #305 (an agent's initial git op landing in the operator's\n"
        "checkout instead of the worktree).\n"
        "\n"
        "If you are an agent: your FIRST shell command should have been\n"
        "  cd <your-worktree-path>\n"
        "Retry after cd'ing into your worktree.\n"
        "\n"
        "If you are the operator and this is intentional, re-run the command\n"
        "with CLAUDE_HOOK_BYPASS=1 in the environment to skip this guard once."
    )
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
