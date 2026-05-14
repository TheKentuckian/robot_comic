# Contributing

This is a personal project repo, kept public for reference and as
a learning artifact. It is **not open for external contributions** —
issues, pull requests, and discussions are closed by design.

If you find something useful here, you're welcome to fork it and
adapt it for your own use.

---

## Issue conventions

These conventions exist for the benefit of the solo maintainer (and any AI
agents working in the repo). They keep the tracker readable and the git log
useful as a project diary.

### Title format

```
<type>(<scope>): <short title>
```

Types: `feat`, `fix`, `chore`, `test`, `refactor`, `docs`

Examples: `feat(personas): add Richard Pryor profile`, `fix(boot): race on daemon ready-poll`

### Labels

| Prefix | Values | Purpose |
|--------|--------|---------|
| `type/` | `bug`, `feature`, `chore`, `research` | What kind of work |
| `priority/` | `p0` (urgent) … `p3` (someday) | When to do it |
| `area/` | `boot`, `voice`, `persona`, `hardware`, `vision`, `infrastructure`, `docs`, … | Which subsystem |
| `phase/` | `1`, `1.5`, `2`, `3`, `4` | Project phase from COMEDIAN_PROJECT.md |

### Body structure

**Bug reports**

1. **Summary** — one sentence describing the defect.
2. **Reproduction** — minimal steps to trigger it; include relevant env vars.
3. **Expected vs actual** — what should happen vs what does.
4. **Acceptance criteria** — checkboxes that define "fixed".

**Feature requests**

1. **Summary** — what capability is being added and why.
2. **Proposed behaviour** — concrete description of the new behaviour.
3. **Acceptance criteria** — checkboxes; each must be verifiable.

### Branch naming

```
feat/<issue#>-<short-slug>
fix/<issue#>-<short-slug>
chore/<issue#>-<short-slug>
```

Example: `feat/63-george-carlin-persona`

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) — same
format as the issue title:

```
<type>(<scope>): <short description>
```

Keep the subject line under 72 characters. Add a body when the "why" needs
more than one line.

### GitHub CLI and remote policy

All issue and PR work is done via the GitHub CLI (`gh`). This repo has a
**single remote — `origin` (TheKentuckian/robot_comic)**. Never fetch from,
push to, or run any `gh` command targeting the upstream fork
(`pollen-robotics/reachy_mini_conversation_app`). Every `gh` invocation must
operate against `--repo TheKentuckian/robot_comic` (or rely on the default
remote, which is `origin`).
