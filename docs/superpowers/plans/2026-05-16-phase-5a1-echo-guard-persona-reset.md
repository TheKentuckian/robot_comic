# Phase 5a.1 — TDD plan: per-persona echo-guard reset

**Spec:** `docs/superpowers/specs/2026-05-16-phase-5a1-echo-guard-persona-reset.md`
**Branch:** `claude/phase-5a-1-echo-guard-persona-reset`

Each task is a single commit: failing test → minimum implementation → green.

## Task 1 — Failing regression test: ElevenLabs handler echo-guard not cleared by composable `apply_personality`

**Test file:** `tests/test_composable_persona_reset.py` (new)
**Body:**

- Construct a real `ElevenLabsTTSResponseHandler` (with `_http` / `_client` mocked).
- Construct a real `ComposableConversationHandler` over a mocked `ComposablePipeline` (`reset_history` side-effect clears the history list, mirroring `test_composable_conversation_handler.py:222-225`).
- Pre-populate `handler._speaking_until = 999.0`, `handler._response_start_ts = 500.0`, `handler._response_audio_bytes = 9600`.
- `await wrapper.apply_personality("rodney")`.
- Assert `handler._speaking_until == 0.0` and the two byte/start fields are likewise reset.

Run:
```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_composable_persona_reset.py -v
```

Expected: **FAIL** (composable path does not clear those fields today).

Commit: `test(phase-5a-1): add failing regression for per-persona echo-guard reset`

## Task 2 — Pinning test: no-op on a handler without echo-guard fields

**Test file:** same `tests/test_composable_persona_reset.py`
**Body:**

- Use a `MagicMock(spec=[])` (no `_speaking_until` etc.) as `_tts_handler`.
- Call `await wrapper.apply_personality("rodney")`.
- Assert no exception, success message returned.

This pins the no-op contract before the implementation lands so the implementation can't drift into raising `AttributeError`.

Expected: **FAIL** for a different reason (today's `apply_personality` doesn't even attempt to touch the TTS handler; the test would pass trivially). After the implementation, the `hasattr` guard makes this a real no-op assertion.

Actually: since the current `apply_personality` doesn't touch `_tts_handler`, this test passes vacuously today. To make it a real pin, the test should assert the call succeeds AND that the wrapper's `_reset_tts_per_session_state` was invoked (even though the handler has no fields). Defer that to Task 5.

For Task 2, just keep the no-op test in this commit — it pins behaviour going forward.

Commit: `test(phase-5a-1): pin no-op behaviour when handler lacks echo-guard fields`

## Task 3 — Implementation: `_reset_tts_per_session_state` helper + call site

**Source file:** `src/robot_comic/composable_conversation_handler.py`
**Changes:**

1. Remove the TODO comment body at lines 172-179 (the `legacy handlers also clear...` block).
2. Replace it with a comment naming the per-session-state reset helper.
3. Add a new `_reset_tts_per_session_state(self)` method on the wrapper (body per spec — guarded `setattr` over three fields).
4. Call `self._reset_tts_per_session_state()` in `apply_personality` after `pipeline.reset_history(keep_system=False)` and before the system-prompt re-seed.

Run:
```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_composable_persona_reset.py -v
```

Expected: **PASS** on both new tests.

Commit: `feat(phase-5a-1): reset wrapped TTS handler echo-guard state on apply_personality`

## Task 4 — Existing-test sanity sweep

Run the broader composable / echo-guard suites to confirm nothing regressed:

```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_composable_conversation_handler.py tests/test_composable_echo_guard.py tests/test_echo_suppression.py -v
```

Expected: all green.

No commit (or trivial fix-and-commit only if something flakes).

## Task 5 — Unit test for the helper-invocation contract

**Test file:** `tests/test_composable_conversation_handler.py`
**Body:** add a new test `test_apply_personality_invokes_tts_reset_helper` that patches `wrapper._reset_tts_per_session_state` and asserts it was called once during `apply_personality`. Mirrors the input-site assertion pattern from `feedback_lifecycle_hooks_not_for_free.md` ("two-test split").

Run:
```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_composable_conversation_handler.py -v
```

Expected: **PASS**.

Commit: `test(phase-5a-1): assert apply_personality invokes the TTS reset helper`

## Task 6 — `PIPELINE_REFACTOR.md` doc nit

Add `**Tracking:** #391` line near the top of the Phase 5 status table at `PIPELINE_REFACTOR.md:26-37`. Audit the "Out of scope for Phase 4" section (line 414+) — no prose change required per the spec's audit (the 4f-retired dials are NOT mentioned there; the brief's "line 388" reference appears to predate the Phase 5 plan being added to the doc, shifting line numbers).

Commit: `docs(phase-5a-1): point Phase 5 status table at tracking epic #391`

## Task 7 — Full local lint/format/type/test sweep

From the repo root:

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/Scripts/python -m mypy --pretty --show-error-codes src/robot_comic/composable_conversation_handler.py
PYTHONPATH=src .venv/Scripts/python -m pytest tests/ -q
```

All four must be green before push. If ruff format wants changes, run `uvx ruff@0.12.0 format .` and commit as `style(phase-5a-1): ruff format`.

## Task 8 — Push (no PR open)

```
git push -u origin claude/phase-5a-1-echo-guard-persona-reset
```

Manager opens the PR after CI is green.

## Suggested PR title and body

**Title:** `feat(phase-5a-1): reset TTS handler echo-guard state on apply_personality`

**Body sketch:**

```markdown
Closes the live TODO at `composable_conversation_handler.py:172-179` plus a tiny `PIPELINE_REFACTOR.md` doc nit. Tracks Phase 5 epic #391.

## What changed

- `ComposableConversationHandler.apply_personality` now clears the wrapped TTS handler's per-session echo-guard accumulators (`_speaking_until`, `_response_start_ts`, `_response_audio_bytes`) after resetting pipeline history. Guarded via `hasattr` so `GeminiTTSResponseHandler` (no echo-guard state) is a clean no-op.
- New `tests/test_composable_persona_reset.py` with two regression tests; new helper-invocation test in `tests/test_composable_conversation_handler.py`.
- `PIPELINE_REFACTOR.md` Phase 5 status table now points at tracking epic #391.

## Why

When the operator switches personas mid-session and the wrapped TTS handler still has a non-zero `_speaking_until` from an in-flight (or recently-completed) playback, `LocalSTTInputMixin._handle_local_stt_event` would keep dropping the operator's next few transcripts to the new persona for the remaining `ECHO_COOLDOWN_MS` window. Persona switch should be a hard cut on listening state, mirroring the existing hard cut on conversation history.

## Audit finding

The TODO body claimed "legacy handlers also clear per-session echo-guard state on persona switch." That overstates what legacy does — `BaseLlamaResponseHandler.apply_personality` and `ElevenLabsTTSResponseHandler.apply_personality` both clear only `_conversation_history`. So this PR is a defense-in-depth improvement, not strict legacy parity. Spec: `docs/superpowers/specs/2026-05-16-phase-5a1-echo-guard-persona-reset.md`.

## Tests

- `test_apply_personality_clears_tts_handler_echo_guard_state` — fails before, passes after (the regression).
- `test_apply_personality_no_op_on_handler_without_echo_guard_fields` — pins the no-op contract for `GeminiTTSResponseHandler`.
- `test_apply_personality_invokes_tts_reset_helper` — pins the call site (two-test split per `feedback_lifecycle_hooks_not_for_free.md`).
```
