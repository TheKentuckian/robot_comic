# Plan — Phase 4d default flip (`FACTORY_PATH=composable`)

**Epic:** #337 (pipeline refactor)
**Sub-phase:** 4d
**Spec:** `docs/superpowers/specs/2026-05-16-phase-4d-default-flip.md`
**Date:** 2026-05-16

## TDD approach

This is too small for genuine "red → green" TDD per task — the test
file and the production constant change have to move together to stay
green at HEAD. But we follow the spirit: the test edits encode the new
behaviour and the production change is the minimum needed to satisfy
them. Two commits total.

## Task 1 — Update default-assertion tests to expect `composable`

**File:** `tests/test_config_factory_path.py`

Edit the five assertions that bake in `"legacy"` as the default /
invalid-fallback:

- `test_constants_defined` — `assert cfg.DEFAULT_FACTORY_PATH == "composable"`.
- `test_normalize_default_when_unset` — assert returns `"composable"` for `None` / `""` / whitespace.
- `test_normalize_invalid_falls_back_to_legacy_with_warning` — rename to `test_normalize_invalid_falls_back_to_composable_with_warning`, assert `result == "composable"`. Keep the `"hybrid"` input and the warning-text check.
- `test_config_field_default` — assert `cfg.config.FACTORY_PATH == "composable"`.
- `test_config_field_invalid_env_falls_back` — assert `cfg.config.FACTORY_PATH == "composable"`.

Run the test file. Expected: the five assertions above fail until
Task 2 lands. The other tests (`test_normalize_known_values`,
`test_config_field_reads_env`) still pass since they explicitly set
the input.

**Commit message:**
```
test(config): expect composable as new FACTORY_PATH default (#337)

Pre-flip the assertions in test_config_factory_path.py for Phase 4d:
unset / blank / invalid REACHY_MINI_FACTORY_PATH now resolves to
"composable". The flip itself lands in the next commit.
```

## Task 2 — Flip the production default + `.env.example`

**Files:**

1. `src/robot_comic/config.py` — single-line change:

   ```diff
   - DEFAULT_FACTORY_PATH = FACTORY_PATH_LEGACY
   + DEFAULT_FACTORY_PATH = FACTORY_PATH_COMPOSABLE
   ```

   No change to `_normalize_factory_path` logic — the warning message
   uses `DEFAULT_FACTORY_PATH` interpolated, so it updates itself.

2. `.env.example` — flip the commented hint and tweak the surrounding
   comment from "Default 'legacy'..." to "Default 'composable'..."; the
   opt-out is `REACHY_MINI_FACTORY_PATH=legacy`.

3. `PIPELINE_REFACTOR.md` — update the 4d row in the status table from
   `⏸ Pending` to `✅ Done | #TBD` (manager will fix the PR number on
   merge).

Run:
- `uvx ruff@0.12.0 check .` from repo root → green.
- `uvx ruff@0.12.0 format --check .` from repo root → green.
- `.venv/Scripts/mypy.exe --pretty src/robot_comic/config.py tests/test_config_factory_path.py` → green.
- `.venv/Scripts/python.exe -m pytest tests/test_config_factory_path.py tests/test_handler_factory_factory_path.py -q` → green (these are the relevant scoped suites).
- `.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py` → green.

**Commit message:**
```
feat(config): flip DEFAULT_FACTORY_PATH to composable (#337)

Phase 4d of the pipeline refactor epic. Unset / blank / invalid
REACHY_MINI_FACTORY_PATH now resolves to "composable" so a fresh
install routes the five migrated triples through ComposablePipeline
+ ComposableConversationHandler by default. Operators can still set
REACHY_MINI_FACTORY_PATH=legacy to fall back until the dial is
retired in 4e/4f.

Spec:  docs/superpowers/specs/2026-05-16-phase-4d-default-flip.md
Plan:  docs/superpowers/plans/2026-05-16-phase-4d-default-flip.md
```

## Verification checklist

- [ ] `uvx ruff@0.12.0 check .` green from repo root.
- [ ] `uvx ruff@0.12.0 format --check .` green from repo root.
- [ ] `mypy --pretty` on changed files green.
- [ ] `pytest tests/ -q` green (modulo the local-only
      `tests/vision/test_local_vision.py` tokenizers/transformers
      collection error).
- [ ] Pre-commit hook passes (no secrets / archival audio in diff).
- [ ] Branch `claude/phase-4d-default-composable` pushed to `origin`.

## Roll-back

Revert the single line in `config.py` (and optionally the `.env.example`
hint). The dial still lives, so no migration is needed — operators
already running with `REACHY_MINI_FACTORY_PATH` set in their env are
unaffected by the flip.
