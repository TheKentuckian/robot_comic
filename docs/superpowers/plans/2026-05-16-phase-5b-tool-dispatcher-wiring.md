# Phase 5b TDD plan — wire `ComposablePipeline.tool_dispatcher`

**Branch:** `claude/phase-5b-tool-dispatcher-wiring`
**Spec:** `docs/superpowers/specs/2026-05-16-phase-5b-tool-dispatcher-wiring.md`
**Predecessor:** `d26caf7` on `main` (Phase 4 epic closed by PR #386).

## Goal

Close the §2.2 latent bug: every composable factory builder that constructs
a tool-enabled pipeline must wire `ComposablePipeline.tool_dispatcher` to
a real dispatcher shim. Bonus: emit a `tool.execute` span around the
dispatch site so the monitor's tool-count column populates again on the
composable path.

## TDD task list — one task per commit

### Task 1 — RED regression test (committed as `88a0913`)

Add `tests/test_phase_5b_tool_dispatcher_wiring.py` with **6 failing
tests** that pin the bug witness:

- 4 parametric assertions that `pipeline.tool_dispatcher is not None` for
  every affected composable triple
  (`(llama|gemini) × (elevenlabs|chatterbox)`).
- 1 routing test — wired dispatcher calls `dispatch_tool_call` and
  returns a string.
- 1 error-surfacing test — `{"error": ...}` round-trips correctly.

The bundled-Gemini triple is excluded (dispatches tools internally).

**Acceptance:** run `pytest tests/test_phase_5b_tool_dispatcher_wiring.py -v`,
**all 6 RED** before any source change. If any test passes before the
fix, STOP — the §2.2 analysis is wrong.

### Task 2 — Wiring fix + span emit (committed as `40343f9`)

In `src/robot_comic/handler_factory.py`:

1. Add module-level helper `_make_tool_dispatcher(host)` that returns a
   closure satisfying `composable_pipeline.ToolDispatcher`:
   - Splits on `_SYSTEM_TOOL_NAMES` (mirrors `ToolCallRoutine.__call__`).
   - System tools go through `dispatch_tool_call_with_manager(..., tool_manager=host.tool_manager)`.
   - Everything else goes through `dispatch_tool_call`.
   - Returns `json.dumps(result)` (legacy parity with
     `llama_base.py:617`).
   - Wraps the dispatch in a `tool.execute` span with `tool.name` /
     `tool.id` / `outcome` attributes (matches
     `background_tool_manager._run_tool` shape).
2. In each `_build_composable_*` helper (llama_elevenlabs,
   llama_chatterbox, gemini_chatterbox, gemini_elevenlabs), add
   `tool_dispatcher=_make_tool_dispatcher(host),` to the
   `ComposablePipeline(...)` call.
3. Leave `_build_composable_gemini_tts` unwired — bundled-Gemini
   dispatches tools inside the adapter (documented in
   `gemini_bundled_llm_adapter.py:17-19`).

Add 2 more tests to `tests/test_phase_5b_tool_dispatcher_wiring.py`:

- `test_dispatcher_emits_tool_execute_span_with_outcome_success`
- `test_dispatcher_emits_tool_execute_span_with_outcome_error`

Both use `InMemorySpanExporter` to capture the emitted span.

**Acceptance:** `pytest tests/test_phase_5b_tool_dispatcher_wiring.py -v`
runs **all 8 GREEN**. `pytest tests/ -q` runs the full suite green
(modulo the `--ignore` + `--deselect` flags called out in the dispatch
instructions for pre-existing Windows quirks).

### Task 3 — Lint + format pass (committed as `9c97428`)

`uvx ruff@0.12.0 check --fix .` + `uvx ruff@0.12.0 format .` — pure
mechanical pass to satisfy CI. No behaviour change.

**Acceptance:** `uvx ruff@0.12.0 check .` and `uvx ruff@0.12.0 format --check .`
both green from repo root.

### Task 4 — Docs (this commit)

Write the spec at
`docs/superpowers/specs/2026-05-16-phase-5b-tool-dispatcher-wiring.md`
and this plan at
`docs/superpowers/plans/2026-05-16-phase-5b-tool-dispatcher-wiring.md`.

**Acceptance:** spec covers §1 bug context, §2 fix design, §3 span emit,
§4 test coverage, §5 diff shape, §6 pre-merge checks, §7 follow-ups.
Plan documents the task-per-commit cadence.

## Done definition

- All 8 phase-5b tests GREEN.
- Full pytest suite GREEN (1720 passed, 10 skipped, 71 deselected for
  known quirks).
- `uvx ruff@0.12.0 check .` + `format --check .` GREEN.
- `mypy --pretty src/robot_comic/handler_factory.py tests/test_phase_5b_tool_dispatcher_wiring.py`
  GREEN.
- Branch pushed to `origin/claude/phase-5b-tool-dispatcher-wiring`.
- PR drafted (body in dispatch return).

## Out of scope

- Background-tool placeholder dispatch (deferred per memo §5.2).
- `robot.tool.duration` histogram (owned by parallel telemetry-housekeeping
  agent — adding a new instrument touches `telemetry.py`).
- Tools-spec source-of-truth migration (the LLM adapters still pull
  tool specs from `deps` rather than the orchestrator's `tools_spec`
  parameter — a separate PR).
- Hardware soak (regression-test surface is sufficient per dispatch
  instructions).
