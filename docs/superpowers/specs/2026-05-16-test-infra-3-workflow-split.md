# Spec: Workflow split + post-merge gate (test-infra-3)

## Context

Today `.github/workflows/pytest.yml` runs the entire suite on every PR
and every push to main. A failing test in a `slow` or `integration`
marker blocks the merge queue. With test-infra-2 in place (the `slow`,
`integration`, `hardware` markers exist), we can split the workflow into
a fast required gate and a full advisory gate.

## Design

### `pytest-fast.yml` (required on PR)

Runs `pytest tests/ -m "not slow and not integration and not hardware"`.

* `-n auto` is already in `addopts` (test-infra-2).
* `timeout-minutes: 2` at the job level.
* A separate "enforce 90 s budget" step parses the duration from the
  pytest output (last line: `1751 passed, ... in 65.09s`) and fails the
  job if the pytest run itself took >90 s. The job-level
  `timeout-minutes` is a hard cap; the 90 s step is a soft regression
  guard.
* Runs on Linux + Windows (matches today's matrix).
* Triggered on `pull_request` and on `push: branches: [main]`. On both
  it is gating (no `continue-on-error`).

### `pytest-full.yml` (advisory on PR; gating on push to main)

Runs `pytest tests/ -m "not hardware"` — i.e. everything except things
that require real ALSA / serial / GPIO devices.

* Triggered on `pull_request` and on `push: branches: [main]`.
* On `pull_request`: `continue-on-error: true` so PR failures are
  advisory only (visible in the checks tab but don't block merge).
* On `push: branches: [main]`: gating + opens a GitHub issue if the
  run fails (notification only, no auto-revert).
* `timeout-minutes: 15` (matches today's full-suite ceiling).

### Why not extend `pytest.yml` in place?

* Two trigger contexts (`pull_request` advisory + `push` gating) for
  the full suite need different `continue-on-error` values per job.
  Cleaner as two workflow files than a job-matrix with conditionals.
* The post-merge issue-creation step is meaningful only on the gating
  run; isolating it in `pytest-full.yml` keeps the fast workflow
  zero-permission (`contents: read` only) and concentrates the
  `issues: write` scope.

### Issue creation on post-merge failure

When `pytest-full` fails on the `push: branches: [main]` trigger, we
open a GitHub issue with:

* Title: `pytest-full failed on main @ <sha-short>`
* Body: workflow run URL, failing job log excerpt, ping
  `@TheKentuckian` (the only repo operator per recent commit authors).

Uses `actions/github-script` so we don't add a custom action dependency.
Required permissions: `contents: read, issues: write` for the gating
job only.

### What this does NOT do

* Does NOT touch branch protection. Operator must manually update
  branch-protection rules on `main` after merge: change the required
  check from `Pytest / pytest (...)` to `pytest-fast / pytest-fast
  (...)`. Called out in PR body.
* Does NOT auto-revert failing main pushes. Notification only — first
  iteration. Operator decides whether to revert, fix-forward, or
  ignore.
* Does NOT change `lint.yml`, `typecheck.yml`, `uv-lock-check.yml` —
  they are already fast and orthogonal.

## Open question

Today the `Pytest` workflow is part of the required checks. After this
PR lands and the operator updates branch protection, the old
`pytest.yml` file becomes dead. **Plan: leave `pytest.yml` in place
in this PR** — operator removes it in a follow-up after branch
protection is updated, to avoid a window where the required check
disappears before its replacement is configured. Called out in PR body.

## Deletion of `pytest.yml`

NOT in this PR. The follow-up is a one-liner once branch protection
flips.
