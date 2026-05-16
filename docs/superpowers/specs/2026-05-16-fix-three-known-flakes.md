# Spec: Fix three known CI flakes (2026-05-16)

## Context

Three tests have been flaking on `main` throughout 2026-05-16. Each has a
distinct root cause. All three are fixable independently and safely.

## Diagnoses

### Flake 1 — `test_run_realtime_session_passes_allocated_session_query` (xdist-only)

**File**: `tests/test_huggingface_realtime.py`

**Failure**:
`AssertionError: assert 'model' not in {'extra_query': {...}, 'model': 'gpt-realtime'}`

**Root cause**: `base_realtime._run_realtime_session` sets
`connect_kwargs["model"]` if `config.MODEL_NAME` is truthy. Sibling tests in
`tests/test_admin_pipeline_3column.py` POST to `/backend_config`, which
triggers `console._persist_env_values` → `config.refresh_runtime_config_from_env()`.
That function mutates `config.MODEL_NAME` directly (no rollback). Within the
same xdist worker process, the leaked `config.MODEL_NAME='gpt-realtime'`
breaks the HF test. Verified by chaining
`pytest tests/test_admin_pipeline_3column.py tests/test_huggingface_realtime.py::test_run_realtime_session_passes_allocated_session_query -p no:xdist`
which reproduces deterministically.

**Fix**: Autouse conftest fixture that snapshots+restores `config` attributes
that `refresh_runtime_config_from_env` is known to mutate.

### Flake 2 — `test_openai_excludes_head_tracking_when_no_head_tracker` (deterministic)

**File**: `tests/test_openai_realtime.py`

**Failure**:
`AssertionError: case 1 failed: a non-head-tracking tool was unexpectedly excluded`
(`fake_tool` is missing from the assembled tool list, and instead the real
tool registry is returned).

**Root cause**: The test patches `core_tools.ALL_TOOL_SPECS` with
`[head_tracking, fake_tool]`. But `get_active_tool_specs` calls
`_initialize_tools()` first, which — if `_TOOLS_INITIALIZED` is still
`False` for this process — overwrites `ALL_TOOL_SPECS` from the real registry,
discarding the patched value. Reproduces in isolation. The companion module
attribute `_TOOLS_INITIALIZED` is the gate that the test fails to also
monkeypatch.

**Fix**: In the test, also `monkeypatch.setattr(ct_mod, "_TOOLS_INITIALIZED", True)`
alongside the existing `ALL_TOOL_SPECS` patch so `_initialize_tools()` becomes
a no-op for the test's duration.

### Flake 3 — `TestHandlerFactoryRealtimeCombinations::test_moonshine_{openai_realtime,hf}_output`

**File**: `tests/test_handler_factory.py`

**Failure**: `NotImplementedError: REACHY_MINI_LLM_BACKEND='gemini' is not yet implemented...`

**Root cause**: Same culprit as Flake 1 — `tests/test_admin_pipeline_3column.py`
POSTs leak `REACHY_MINI_LLM_BACKEND` into `os.environ` (via
`console._persist_env_values` → `os.environ[env_name] = value`) and mutate
`config.LLM_BACKEND` directly (via `_persist_local_stt_settings`). The
handler factory tests don't patch `config.LLM_BACKEND` for the realtime hybrids,
so `getattr(config, "LLM_BACKEND", ...)` returns the leaked `'gemini'`,
falling into the unsupported-combination arm.

**Fix**: Same conftest-level autouse fixture as Flake 1. The fixture also
snapshots `os.environ` for the well-known leak-prone keys and restores them.

## Fixes summary

1. `tests/conftest.py` — add a session-safe autouse fixture that snapshots and
   restores: the `REACHY_MINI_*` and `MODEL_NAME` env vars known to be mutated
   by `console._persist_env_values`, plus the `config` singleton attributes
   that `refresh_runtime_config_from_env` overwrites.
2. `tests/test_openai_realtime.py::test_openai_excludes_head_tracking_when_no_head_tracker`
   — also monkeypatch `_TOOLS_INITIALIZED = True` so the `ALL_TOOL_SPECS` patch
   survives the lazy-init guard.

## Out of scope

- Refactoring `core_tools._initialize_tools` to honour patched module state
  (e.g. snapshotting at first init, or guarding overwrites).
- Refactoring `console._persist_env_values` to use a single setter that tests
  can patch.
- Adding broad pytest hooks beyond `tests/conftest.py`.
- Disabling xdist or adding serial markers.

## Verification

- Each failing test in isolation (`-p no:xdist`).
- Each failing test under xdist (`-n auto`).
- Full suite under xdist, three runs back-to-back.
