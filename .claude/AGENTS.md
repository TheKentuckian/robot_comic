# Agent self-check directive

When dispatched as a subagent with `isolation: worktree`, your FIRST shell
command MUST be a working-directory canary:

```bash
cd "<your-worktree-path>" && pwd && git rev-parse --show-toplevel
```

Verify both outputs match the worktree path you were given in the dispatch
prompt. If they don't, STOP and report the mismatch to the parent agent
**before** running any git or Edit/Write operation. This catches the
CWD-leak pattern from issue #305 at the bootstrap point, before edits can
land in the wrong checkout.

A repo-local `PreToolUse:Bash` hook
(`.claude/hooks/check_worktree_cwd.py`, wired in `.claude/settings.json`)
also blocks `git checkout -b NEW`, `git switch -c NEW`, and
`git worktree add` if those run from the main-repo CWD while an active
agent worktree exists. The hook is the safety net; the self-check above
is the canary.

The operator can bypass the hook for a single command with
`CLAUDE_HOOK_BYPASS=1` in the environment.
