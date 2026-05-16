# Greet — terminal `no_subject` when sweep is disabled

**Date:** 2026-05-16
**Scope:** tool-side fix only — no persona prompt edits, no orchestrator changes
**Branch:** `claude/fix-greet-terminal-when-sweep-disabled`
**Status:** spec → implementation

## Problem

During hardware validation today (2026-05-16) with the sweep kill-switch in
place (`REACHY_MINI_GREET_SWEEP_DISABLED=1`, see #264 for the chassis-safety
reason), the Don Rickles Gemini persona looped `greet action="scan"` four
times in a single user turn because every call returned the bare
`{"no_subject": True}` dict. The LLM read that as "the scan didn't find the
person yet, try again":

```
17:06:43  stt.infer "Hello." (1.7s)
17:06:49  greet success (5.0s) → no_subject
17:06:56  greet success (5.2s) → no_subject  ← 2nd
17:07:02  greet success (5.0s) → no_subject  ← 3rd
17:07:08  greet success (5.0s) → no_subject  ← 4th
17:07:10  tts.synthesize (finally spoke)
```

27 seconds from "Hello." to the first spoken response, almost entirely tool
latency. The orchestrator's `DEFAULT_MAX_TOOL_ROUNDS = 8`
(`composable_pipeline.py:99`) would have allowed up to 8 such loops. The
conversation feels broken on hardware.

## Root cause

When `REACHY_MINI_GREET_SWEEP_DISABLED` is truthy, `Greet._scan`
short-circuits the head sweep and returns `{"no_subject": True}` (see
`src/robot_comic/tools/greet.py:334-343`). With the sweep disabled, the only
way the result could change between two calls within the same user turn is
if a new face wandered head-on into frame in the ~5s the previous call took
— effectively never. The return value is indistinguishable from the
"sweep-completed-and-still-found-nothing" case, where retry is at least
plausible. The LLM has no way to tell the two apart, and Gemini chooses to
retry.

## Decision: tool-side fix

We fix the tool's return value, not the persona prompt or the orchestrator:

- **Persona-side fix** ("tell Don Rickles never to call greet more than once
  per turn") is fragile — it relies on a specific persona/LLM combination
  reading and obeying the prompt. It also has to be repeated in every
  profile's `instructions.txt` (currently 8 personas reference `greet`).
- **Orchestrator-side fix** ("same tool repeated N times in one turn → break
  and force text response") is a broader cross-cutting change that interacts
  with legitimate tool repetition patterns (e.g. consecutive `play_emotion`
  calls). The operator wants to try the targeted tool-side fix first.

The tool itself knows whether its result is a transient miss or a
configuration-induced certainty. Making the return value self-describing is
the smallest, most local fix that helps every LLM backend (current Gemini,
future Llama / Hermes, etc.) without per-persona prompt tuning.

## Audit of all `no_subject` sites

Grep across the file shows two return sites in `greet.py`:

1. **Line 343 — sweep-disabled branch (THIS BUG)**
   Kill-switch is set. Initial poll window expired with no face. Sweep is
   skipped. Returns `{"no_subject": True}`.
   **Outcome is fixed by configuration.** Calling again in the same turn
   cannot produce a different result barring an unrelated person walking
   into the frame in the ~5s window. → terminalize.

2. **Line 386 — full sweep completed**
   Sweep ran all 4 positions (`left`, `up`, `right`, `front`) and found
   nothing. Returns `{"no_subject": True}`.
   **Retryable in principle.** The robot has physically scanned the room
   and confirmed empty, so a second call still has a chance of latching
   onto someone who just entered (~25s window from the previous attempt's
   sweep + poll). Per the dispatch instructions, this site keeps the
   existing semantics.

The fix is therefore narrowly scoped to site (1).

There is also one site in `roast.py:92` returning `{"no_subject": True}` —
out of scope for this fix; the persona uses it differently (one-shot, after
`greet` has already confirmed a subject).

## Chosen return shape

Enrich the existing dict at site (1) so existing prompt rules like
`If no_subject: true — the room is empty` keep working unchanged. New
fields are additive:

```python
return {
    "no_subject": True,            # unchanged — backwards compatible
    "sweep_disabled": True,        # NEW — names the configuration cause
    "retry_hint": "do_not_retry",  # NEW — machine-readable directive
    "note": (                       # NEW — human-readable; this is what
        "Head sweep is disabled by configuration "
        "(REACHY_MINI_GREET_SWEEP_DISABLED). The scan only checked the "
        "head-on view and saw nobody. Calling greet again this turn will "
        "produce the same result — do not retry. Improvise as if the "
        "room is empty."
    ),
}
```

Site (2) — the genuine sweep-completed-empty case — stays as
`{"no_subject": True}` (unchanged). The orchestrator and persona contracts
are untouched.

## Why this is enough

The LLM-readable tool response is the JSON encoding of the result dict
(see `core_tools.py:330` → backend-side formatting). Both Gemini and
OpenAI/HF backends serialize the dict to a JSON string the LLM reads. The
new `note` field describes the situation in natural language; the
`retry_hint` field gives any future programmatic guard (orchestrator or
backend) a stable key to read; the `sweep_disabled` flag names the
configuration cause for log diagnosis. Gemini in particular reads field
names along with values, so `retry_hint: "do_not_retry"` plus a sentence
saying "do not retry" should defuse the retry loop without needing prompt
tuning.

## Test plan

All tests go in the existing test files; we do not add new files.

1. **`tests/tools/test_greet.py`** — extend
   `test_scan_no_face_after_full_sweep` is unaffected (sweep path). Add a
   new test `test_scan_sweep_disabled_returns_terminal_no_subject` that:
   - Sets `REACHY_MINI_GREET_SWEEP_DISABLED=1` and
     `REACHY_MINI_GREET_SCAN_WAIT_S=0.0` via `monkeypatch.setenv`.
   - Mocks `_detect_face_with_scores` to return `(False, [])`.
   - Asserts the result contains `no_subject=True`, `sweep_disabled=True`,
     `retry_hint="do_not_retry"`, and a non-empty `note`.

2. **`tests/test_greet_scan_diagnostics.py`** — extend the existing
   `test_scan_emits_info_summary_on_no_subject` and
   `test_scan_emits_debug_detection_list_even_when_empty` checks. Both set
   the sweep-disabled env var; relax their `assert result == {"no_subject":
   True}` equality assertion to `assert result["no_subject"] is True` so
   the added fields don't break them. (Same for
   `test_scan_logs_when_poll_frame_is_none`.)

3. **`tests/test_greet_scan.py`** — `test_scan_waits_up_to_configured_
   window_before_giving_up` does NOT set the sweep-disabled env, so its
   `assert result == {"no_subject": True}` strict equality remains valid
   (sweep path returns the unchanged dict). No change needed.

4. **Whole-repo `uvx ruff@0.12.0 check .` + `format --check .`** must
   pass.

5. **`pytest tests/ -q`** must pass modulo the known main-branch flakes
   (handler_factory, openai_realtime, huggingface_realtime,
   gemini_turn_buffers — re-run if hit).

## Files touched

- `src/robot_comic/tools/greet.py` — site (1) return value only
- `tests/tools/test_greet.py` — add one positive test
- `tests/test_greet_scan_diagnostics.py` — relax 3 equality assertions

Estimated diff: ~60-80 LOC including test changes. Well inside the
~100-200 LOC budget.

## Out of scope (explicitly)

- Persona prompt edits (`profiles/*/instructions.txt`)
- Orchestrator-level retry caps
  (`composable_pipeline.py::_run_llm_loop_and_speak`)
- Sweep-completed-empty case at line 386
- `roast.py` `no_subject` return at line 92
- The head sweep itself (#264 / #308)
