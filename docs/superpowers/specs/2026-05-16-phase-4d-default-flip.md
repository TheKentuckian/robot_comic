# Phase 4d — Flip default `FACTORY_PATH` to `composable`

**Epic:** #337 (pipeline refactor)
**Sub-phase:** 4d
**Status:** spec
**Date:** 2026-05-16

## Goal

Flip the default value of `REACHY_MINI_FACTORY_PATH` from `legacy` to
`composable` so a fresh checkout/install routes the migrated triples
through `ComposableConversationHandler` + `ComposablePipeline` by
default. Operators who need to fall back can still set
`REACHY_MINI_FACTORY_PATH=legacy` until the dial is retired in 4e/4f.

## Why now

- All five composable triples are live (4c.1 – 4c.5 merged).
- All five composable lifecycle hooks landed before the flip (per the
  "Composable lifecycle hooks" table in `PIPELINE_REFACTOR.md`).
- 4c-tris was skipped (Option B memo) — the two remaining hybrid
  realtime handlers stay on the legacy path until 4e and are reached
  via `BACKEND_PROVIDER`, not the migrated triples.
- The dispatch matrix in `handler_factory.py` only routes the migrated
  triples down the composable branch; unmigrated paths fall through to
  the legacy concrete handlers regardless of `FACTORY_PATH`. Flipping
  the default therefore changes runtime behaviour only for the five
  migrated triples — which is the intended cutover.

## Change set

This is intentionally tiny — three production-code/config edits and
one test edit.

| File | Change |
|------|--------|
| `src/robot_comic/config.py` | `DEFAULT_FACTORY_PATH = FACTORY_PATH_LEGACY` → `DEFAULT_FACTORY_PATH = FACTORY_PATH_COMPOSABLE`. |
| `.env.example` | Update the commented hint `# REACHY_MINI_FACTORY_PATH=legacy` → `# REACHY_MINI_FACTORY_PATH=composable`, and tweak the surrounding comment to note that `legacy` is the opt-out. |
| `tests/test_config_factory_path.py` | Update the three assertions that depend on the default value (`test_constants_defined`, `test_normalize_default_when_unset`, `test_config_field_default`) and the invalid-fallback assertions (`test_normalize_invalid_falls_back_to_legacy_with_warning`, `test_config_field_invalid_env_falls_back`) to expect `composable` as the new fallback. Rename the warning test accordingly. |
| `PIPELINE_REFACTOR.md` | Update the 4d row in the status table from `⏸ Pending` to `✅ Done | #<PR>` (placeholder `#TBD` is fine — manager fills on merge). |

Everything else is untouched. The other factory-path tests
(`test_handler_factory_factory_path.py`) explicitly `monkeypatch.setattr`
`config.FACTORY_PATH` to the value under test and do **not** depend on
the default, so they need no change. Their docstrings that say
"`FACTORY_PATH=legacy` (default)" are slightly misleading after the
flip, but they remain factually correct under the explicit-set they
perform — and rewording them is out of scope for the minimal-surgical
PR. (4e will rewrite this whole file anyway when the legacy handlers
are deleted.)

## Risk

- Low. Every code path the flip affects is already exercised by the
  composable test suite under `FACTORY_PATH=composable`. The flip just
  makes that the path you get without an env var.
- The bundled realtime handlers (HuggingFace / OpenAI Realtime / Gemini
  Live / LocalSTT*Realtime) ignore `FACTORY_PATH` entirely — verified
  by `test_bundled_realtime_modes_ignore_factory_path` in
  `test_handler_factory_factory_path.py`.
- Operator can roll back by setting `REACHY_MINI_FACTORY_PATH=legacy`
  in `.env` without any code change.

## Out of scope

- Deleting legacy concrete handlers (that's 4e, gated on operator
  soak green-light).
- Retiring the `FACTORY_PATH` dial itself (4e/4f).
- Touching the "default" wording in `test_handler_factory_factory_path.py`
  docstrings — left for 4e's wholesale rewrite.
- Hardware soak — the operator has explicitly lifted the hard pause
  for this PR; merge on green CI.
