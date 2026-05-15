**A green `push`-event CI run + a red `pull_request`-event CI run on the same SHA means main is broken, not the PR.**

GitHub's `lint.yml` (and most workflows) run on both `push` and `pull_request`. They use different `github.ref` values, so they land in different concurrency groups and BOTH run to completion.

- `push` event → runs against your branch tip in isolation.
- `pull_request` event → runs against the **merge commit** (`refs/pull/N/merge` = your branch merged into main).

If main is red (e.g. someone merged a PR with failing CI), the pull_request-event run picks up the breakage and fails even though your branch is clean. Diagnostic: open the failed job's workflow URL and check whether the failure cites a file you didn't touch.

Today (2026-05-15): PR #354 (motion_safety) was merged with both ruff CI runs failing. The unformatted `src/robot_comic/motion_safety.py:158-163` then broke every subsequent PR's pull_request-event ruff. Fix was PR #356 — single-line `ruff format` collapse.

Mitigation: before opening a feature PR, sanity-check that main's ruff/mypy/pytest CI is green on the latest commit. If not, open a tiny fix PR first, merge it, then open the feature PR.
