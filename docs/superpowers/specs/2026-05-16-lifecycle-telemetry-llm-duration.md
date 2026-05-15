# Lifecycle Hook #2 — `telemetry.record_llm_duration` on the composable path

**Branch:** `claude/lifecycle-telemetry-llm-duration`
**Epic:** #337 (pipeline refactor) — Deferred lifecycle hooks
**Date:** 2026-05-16
**Author:** sub-agent (manager-driven)

## Background — why the original "wrap timing in the adapter" plan is correct, but telemetry is currently bypassed

`PIPELINE_REFACTOR.md` line 312 lists `telemetry.record_llm_duration` as a
deferred lifecycle hook and proposes the new home as "`LlamaLLMAdapter.chat`
/ `GeminiLLMAdapter.chat` (wrap timing in the adapter)". Like Hook #1
(echo-guard), this implies the relocation is conceptually clean. It does
not say the telemetry is preserved through delegation. Investigation
confirms it is not.

### Where `record_llm_duration` lives today (legacy paths)

`git grep "telemetry.record_llm_duration" src/` returns four legacy write
sites:

| File | Line | Method | Wraps |
|------|------|--------|-------|
| `src/robot_comic/llama_base.py` | 574 | `_run_response_loop` (round 1) | `_stream_response_and_synthesize` |
| `src/robot_comic/llama_base.py` | 640 | `_run_response_loop` (follow-up rounds 2..N) | `_call_llm` |
| `src/robot_comic/elevenlabs_tts.py` | 799 | `_run_llm_with_tools` (error path) | `_llm_generate_with_backoff` |
| `src/robot_comic/elevenlabs_tts.py` | 829 | `_run_llm_with_tools` (success path) | `_llm_generate_with_backoff` |
| `src/robot_comic/base_realtime.py` | 892 | `_run_realtime_response` | realtime LLM call (out of scope) |

The `base_realtime.py` site stays — Phase 4c.5 keeps the realtime path
legacy-only.

### What the three composable LLM adapters call

- `LlamaLLMAdapter.chat()` → `handler._call_llm(extra_messages=messages)`
  directly. **Bypasses** `_run_response_loop`, so neither line 574 nor 640
  fires.
- `GeminiLLMAdapter.chat()` → `handler._call_llm(extra_messages=messages)`
  directly (gemini-text's `_call_llm` overrides the llama base). Same
  bypass.
- `GeminiBundledLLMAdapter.chat()` → `handler._run_llm_with_tools()` on
  `GeminiTTSResponseHandler`. That method, in `gemini_tts.py:469`, never
  calls `telemetry.record_llm_duration` at all — only the
  `elevenlabs_tts.py` cousin does. So legacy bundled-Gemini already lacks
  this telemetry, and the adapter inherits the gap.

### Conclusion

The "wrap timing in the adapter" plan is the correct fix shape. There is
no path through delegation that preserves the legacy write — the legacy
write site lives in a method (`_run_response_loop`) that the composable
path bypasses entirely. **This PR adds the timing wrap inside each
adapter's `chat()` method.**

For the bundled-Gemini case, this PR also adds telemetry that legacy never
had — which is a small upgrade rather than a regression, and is consistent
with the other two adapters so dashboards have one consistent surface.

## Decision — fix shape

Wrap timing inside the three LLM adapters' `chat()` methods using
`time.perf_counter()` brackets and a `try / finally` that fires
`telemetry.record_llm_duration` on every code path (success, exception,
empty response). Mirror the legacy attribute shape:

```python
{"gen_ai.system": <system>, "gen_ai.operation.name": "chat"}
```

`<system>` values:

| Adapter | `gen_ai.system` | Rationale |
|---------|----------------|-----------|
| `LlamaLLMAdapter` | `"llama_cpp"` | Matches legacy `llama_base.py:574, 640` exactly. |
| `GeminiLLMAdapter` | `"gemini"` | Matches `_LLM_SYSTEM = "gemini"` in `elevenlabs_tts.py:244`. Legacy gemini-text's `_run_response_loop` emits `"llama_cpp"` by inheritance accident — fixing on the way through, with the manager's blessing implied by "wrap timing in the adapter" in the doc table. |
| `GeminiBundledLLMAdapter` | `"gemini"` | New telemetry (legacy bundled never emitted this). Consistent with `GeminiLLMAdapter`. |

The legacy write sites in `llama_base.py:574 / 640` and
`elevenlabs_tts.py:799 / 829` remain untouched in this PR. They cover the
legacy `_run_response_loop` / `_run_llm_with_tools` paths, which are still
the production path until Phase 4d default-flip. Removing them is part of
Phase 4e (legacy retirement), out of scope here.

After Phase 4d, all production LLM calls flow through one of the three
adapters, so the adapter-side timing is the single source of truth. The
legacy sites become dead code that 4e deletes.

## Scope

| File | Change |
|------|--------|
| `src/robot_comic/adapters/llama_llm_adapter.py` | `chat()` brackets `await self._handler._call_llm(...)` with `time.perf_counter()` and calls `telemetry.record_llm_duration` in a `finally` (so exception paths still record). `gen_ai.system="llama_cpp"`. |
| `src/robot_comic/adapters/gemini_llm_adapter.py` | Same shape; `gen_ai.system="gemini"`. |
| `src/robot_comic/adapters/gemini_bundled_llm_adapter.py` | Same shape; `gen_ai.system="gemini"`. Wraps `_run_llm_with_tools()`. |
| `tests/adapters/test_llama_llm_adapter.py` | New tests: telemetry fired on success path, fired on exception path, attrs correct. |
| `tests/adapters/test_gemini_llm_adapter.py` | Same regression coverage. |
| `tests/adapters/test_gemini_bundled_llm_adapter.py` | Same regression coverage. |

## Files NOT touched

- `src/robot_comic/llama_base.py`, `elevenlabs_tts.py`, `base_realtime.py`
  — legacy write sites stay until Phase 4e.
- `src/robot_comic/composable_pipeline.py`,
  `composable_conversation_handler.py` — telemetry stays in the LLM
  surface, not the orchestrator. Other LLM adapters (future) get the same
  treatment automatically because the orchestrator never knows which
  backend emits.
- `src/robot_comic/telemetry.py` — the `record_llm_duration` API and its
  histogram already exist; no changes.
- Adapter Protocols, `ConversationHandler` ABC, `handler_factory.py` — no
  surface changes.

## Why timing lives in the adapter (not in `_call_llm` or
`_run_llm_with_tools`)

Two reasons:

1. The legacy `_run_response_loop` and `_run_llm_with_tools` deliberately
   keep their telemetry scoped to the LLM round-trip — not the entire
   loop. Pushing telemetry up to the adapter's `chat()` keeps that scope
   (one `chat()` call == one LLM round-trip from the orchestrator's
   perspective).
2. Putting telemetry inside `_call_llm` (the deepest shared site) would
   make the timing visible in *both* legacy and composable paths
   simultaneously, doubling the metric in legacy mode (where
   `_run_response_loop` already wraps `_call_llm` with its own timing).
   Adapter-only is the cleaner blast radius — legacy gets exactly its
   legacy timing, composable gets exactly the adapter timing, never both.

## Acceptance criteria

- Each of the three adapter test files gets at least two new tests:
  - One assertion that `telemetry.record_llm_duration` is called on the
    success path with `duration_s > 0` and the expected attrs dict.
  - One assertion that it still fires on the exception path (legacy
    parity — the legacy `finally` block also records).
- All existing adapter tests still pass.
- `uvx ruff@0.12.0 check` and `format --check` green from repo root.
- `.venv/bin/mypy --pretty <changed files>` green.
- `.venv/bin/pytest tests/ -q` green.
- Adapter-side timing exists in exactly one place per LLM adapter — never
  doubled with the legacy sites (which remain on the legacy path only).

## Out of scope

- Removing the legacy write sites in `llama_base.py` / `elevenlabs_tts.py`
  — Phase 4e cleanup.
- Adding richer attributes (`gen_ai.request.model`, token counts) — the
  legacy attrs are minimal and this PR mirrors them. A follow-up can
  expand once dashboards prove they need it.
- Wiring `record_llm_duration` into `base_realtime.py` or the realtime
  loops — they're not on the composable path.
- Reconciling the `"llama_cpp"` vs `"gemini"` mismatch in the legacy
  gemini-text path. That's a legacy-side bug; this PR ships
  semantically-correct values on the *new* surface. Phase 4e legacy
  deletion is when the inconsistency disappears.
