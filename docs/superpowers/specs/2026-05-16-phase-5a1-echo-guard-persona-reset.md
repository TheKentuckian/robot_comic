# Phase 5a.1 — Per-persona echo-guard reset on `apply_personality`

**Branch:** `claude/phase-5a-1-echo-guard-persona-reset`
**Epic:** #391 (Phase 5)
**Date:** 2026-05-16
**Author:** sub-agent (Phase 5 manager-driven)
**Predecessors:** Lifecycle Hook #1 (PR #372) — per-frame `_speaking_until` write
**Spec template:** mirrors `docs/superpowers/specs/2026-05-15-lifecycle-echo-guard-fix.md`

This sub-phase ships two small things in one PR:

1. **Wire per-persona reset of echo-guard accumulators on `apply_personality`** (the live TODO on `composable_conversation_handler.py:172-179`).
2. **Doc nit on `PIPELINE_REFACTOR.md`** — refresh the stale "Out of scope" bullet that still implies Phase 5 inherits the `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` retirement (Phase 4f did the work in PR #381). Add a "Tracking: #391" pointer near the Phase 5 status table.

## Audit results — what per-session state lives on each surviving TTS handler?

Surviving response-handler bases used as adapter targets post-4e:

| Handler class | File | Per-session state |
|---|---|---|
| `BaseLlamaResponseHandler` (subclassed by `LlamaElevenLabsTTSResponseHandler`, `ChatterboxTTSResponseHandler`, `GeminiTextResponseHandler` and its `Chatterbox`/`ElevenLabs` diamond subclasses) | `llama_base.py:64`, fields at `:115-117` | `_speaking_until`, `_response_start_ts`, `_response_audio_bytes` |
| `ElevenLabsTTSResponseHandler` | `elevenlabs_tts.py:231`, fields at `:316-319` | `_speaking_until`, `_response_start_ts`, `_response_audio_bytes`, `_is_responding`, `_dispatch_in_flight` |
| `ChatterboxTTSResponseHandler` (inherits from `BaseLlamaResponseHandler`) | `chatterbox_tts.py:47` | inherits the three echo-guard accumulators from the base |
| `GeminiTTSResponseHandler` | `gemini_tts.py:194` | **none** — bundled Gemini TTS has no echo-guard state today (confirmed by `Grep` of `_speaking_until|_response_audio_bytes|_response_start_ts` in `gemini_tts.py` → no matches) |

### What does the legacy `apply_personality` actually clear today?

| Handler | Body | What it clears |
|---|---|---|
| `BaseLlamaResponseHandler.apply_personality` (`llama_base.py:250-259`) | `set_custom_profile(...) ; self._conversation_history.clear()` | history only |
| `ElevenLabsTTSResponseHandler.apply_personality` (`elevenlabs_tts.py:489-497`) | same | history only |
| `GeminiTTSResponseHandler.apply_personality` (`gemini_tts.py:325-333`) | same | history only |
| `BaseRealtimeHandler.apply_personality` (`base_realtime.py:322-…`) | profile + session.update — no echo-guard state to clear (no `_speaking_until` field on that side) | n/a |

**Finding:** the legacy code path **does not clear `_speaking_until` / `_response_audio_bytes` / `_response_start_ts` on persona switch either.** The TODO on `composable_conversation_handler.py:158-166` ("legacy handlers also clear per-session echo-guard state on persona switch") **overstates** what legacy does. Strict legacy parity is already achieved by the wrapper's `pipeline.reset_history(keep_system=False)`.

But the TODO's *intent* (and the Phase 5 exploration memo §3.1 follow-up) is correct as a defense-in-depth improvement: when the operator switches personas mid-session, if a long TTS playback is in flight on the wrapped handler (or just ended within the `ECHO_COOLDOWN_MS` window), `LocalSTTInputMixin._handle_local_stt_event` (`local_stt_realtime.py:619-622`) will keep dropping transcripts off `_speaking_until` for the remaining cooldown — even though those transcripts are now intended for the *new* persona's listening window.

The composable wrapper resets history + re-seeds the system prompt as a deliberate hard cut between personas; the echo guard should follow the same hard-cut semantics.

### "For free" verification — does any delegation path already clear this?

Audited:

- `ComposableConversationHandler.apply_personality` (`composable_conversation_handler.py:170-187`) — does NOT touch `_tts_handler`'s echo-guard fields. Only resets pipeline history.
- `ComposablePipeline.reset_history` — operates on `_conversation_history`. No TTS handler access.
- TTS adapters (`ElevenLabsTTSAdapter`, `ChatterboxTTSAdapter`, `GeminiTTSAdapter`) — none expose a `reset_per_session_state` method.

**Verified:** no existing delegation path clears `_speaking_until` on persona switch. A real fix is required. This matches the Phase 4 pattern (5/5 lifecycle hooks needed code fixes — see `feedback_lifecycle_hooks_not_for_free.md`).

## Proposed shape

Mirror the existing `_clear_queue` mirroring pattern on the wrapper (`composable_conversation_handler.py:74-94`): the wrapper holds the legacy TTS handler by reference for exactly this purpose, and per-session state lives on that handler. The fix touches **only** `composable_conversation_handler.py` plus a new test file.

### Change

In `ComposableConversationHandler.apply_personality`, after `pipeline.reset_history(keep_system=False)` and *before* re-seeding the system message, call a private `_reset_tts_per_session_state()` helper that defensively clears the echo-guard accumulators on `self._tts_handler` using `setattr`-with-guard semantics (so handlers that lack the fields — `GeminiTTSResponseHandler` — are no-ops).

```python
def _reset_tts_per_session_state(self) -> None:
    """Defensively clear per-session echo-guard accumulators on the wrapped TTS handler.

    Persona switch is a hard cut: any in-flight TTS playback's
    ``_speaking_until`` window must not bleed into the new persona's
    listening window, or LocalSTTInputMixin would keep dropping the
    operator's first few transcripts to the new persona.

    Handlers without echo-guard fields (e.g. ``GeminiTTSResponseHandler``)
    are no-ops via guarded ``setattr``.
    """
    handler = getattr(self, "_tts_handler", None)
    if handler is None:
        return
    for field, value in (
        ("_speaking_until", 0.0),
        ("_response_start_ts", 0.0),
        ("_response_audio_bytes", 0),
    ):
        if hasattr(handler, field):
            setattr(handler, field, value)
```

### Why on the wrapper, not on each adapter?

The brief floated "add a `reset_per_session_state()` method on the TTS adapter." That's the cleaner long-term shape (and lines up with Phase 5c's voice/personality redesign). For 5a.1, putting the logic on the wrapper:

1. Keeps the diff to a single file (plus tests).
2. Reuses the wrapper's existing access pattern to `_tts_handler` (already used by `_clear_queue` setter, voice forwarders, etc.).
3. Doesn't churn the `TTSBackend` Protocol or the three adapters — leaves that for 5c where the Protocol-level voice/personality contract is being redesigned anyway.
4. Aligns with the operator's "5a is small TODO cleanup" framing (Phase 5 exploration memo §4.5a).

If Phase 5c later moves `apply_personality` onto `ComposablePipeline` and the wrapper stops holding `_tts_handler` by reference, this helper migrates with it (becomes a method on the future `TTSBackend` Protocol extension, default no-op).

### Why NOT clear `_is_responding` / `_dispatch_in_flight`?

Those guards live on `ElevenLabsTTSResponseHandler` but are only checked from `ElevenLabsTTSResponseHandler._dispatch_completed_transcript` (`:546`) — a legacy dispatch path that the composable pipeline bypasses entirely. Clearing them would be a no-op on the composable path and could mask bugs if the legacy dispatch ever re-engages. Leave them alone.

### Why NOT clear canned-opener accumulators?

The canned-opener-path resets at `elevenlabs_tts.py:427-428` and `llama_base.py:183-184` run as part of `_send_startup_trigger`. Persona switch happens post-boot, after the opener has long completed. The same `_response_audio_bytes` / `_response_start_ts` accumulators are what the echo-guard reset above already clears, so this is one and the same.

## Regression tests

Two tests in a new `tests/test_composable_persona_reset.py` file:

1. **`test_apply_personality_clears_tts_handler_echo_guard_state`** — wraps a real `ElevenLabsTTSResponseHandler`, sets the three accumulators to non-zero values (simulating mid-playback), calls `await wrapper.apply_personality(profile)`, asserts all three are reset. Fails before the fix; passes after.
2. **`test_apply_personality_no_op_on_handler_without_echo_guard_fields`** — same scaffolding but wraps `GeminiTTSResponseHandler` (no `_speaking_until` fields). Asserts `apply_personality` succeeds without raising — the `hasattr`-guarded `setattr` must skip cleanly. Passes both before and after — pins the no-op contract.

A third unit-level test in `tests/test_composable_conversation_handler.py` asserts the helper is invoked exactly once per `apply_personality` call (so future refactors that drop the call get caught):

3. **`test_apply_personality_invokes_tts_reset_helper`** — patches `_reset_tts_per_session_state`, calls `apply_personality`, asserts it was called.

This is the Phase 4 "two-test diagnostic split" pattern (`feedback_lifecycle_hooks_not_for_free.md`): test #1 is the regression on the field of interest; test #3 is the input-site assertion that the call actually happens.

## PIPELINE_REFACTOR.md stale-bullet update

The brief flags `PIPELINE_REFACTOR.md:388` as stale. Checking the file:

- Line 414: `## Out of scope for Phase 4 (the whole epic, not just 4c)` heading is the section the brief refers to. The bullets at lines 416-420 are general carve-outs (bundled-realtime handlers stay as-is, `LocalSTTInputMixin` survives, etc.) — they don't reference `BACKEND_PROVIDER`.
- The dial retirement is its own sub-phase (`PIPELINE_REFACTOR.md:292-323` — "Sub-phase 4f — Retire BACKEND_PROVIDER / LOCAL_STT_RESPONSE_BACKEND") and is correctly marked ✅ in the status table at line 24.

The Phase 5 exploration memo's appendix (`docs/superpowers/specs/2026-05-16-phase-5-exploration.md:833-841`) makes the same observation and recommends a one-line edit. The actually-stale prose lives in the exploration memo's own narrative (§3 of the original Phase 4 exploration memo, referenced as a prediction) — but that's a historical document. The Phase 5 manager's note is to make sure the *epic-tracking* doc (`PIPELINE_REFACTOR.md`) is unambiguous.

**Fix:** add `Tracking: #391` to the Phase 5 status table header in `PIPELINE_REFACTOR.md` so sub-PRs reference the right epic. Audit the "Out of scope" section for any remaining language that hints the dial is still pending — none found at a careful re-read — so no prose change required there.

## Acceptance criteria

- Two new tests in `tests/test_composable_persona_reset.py` pass with the fix; test #1 fails without it.
- One new test added to `tests/test_composable_conversation_handler.py` covering the helper-invocation contract.
- All existing tests in `tests/test_composable_conversation_handler.py`, `tests/test_composable_echo_guard.py`, `tests/test_echo_suppression.py` still pass.
- `composable_conversation_handler.py:172-179` TODO body removed; replaced with a comment pointing at the helper.
- `uvx ruff@0.12.0 check .` and `format --check .` green from repo root.
- `mypy --pretty` green on the changed file.
- `PIPELINE_REFACTOR.md` has `Tracking: #391` near the Phase 5 status table.

## Files NOT touched

- `src/robot_comic/elevenlabs_tts.py`, `llama_base.py`, `chatterbox_tts.py`, `gemini_tts.py` — legacy handler internals untouched (5a is not a refactor).
- `src/robot_comic/composable_pipeline.py` — unchanged.
- `TTSBackend` Protocol in `backends.py` — unchanged; Protocol churn is 5c's territory.
- `src/robot_comic/adapters/*` — unchanged; no `reset_per_session_state` method added to adapters in this PR.
- `src/robot_comic/base_realtime.py` — bundled-realtime handlers don't have echo-guard state; out of scope.

## Out of scope

- Wiring echo-guard state into `gemini_tts.py` (it's a pre-existing gap, not regressed by this PR).
- Moving `apply_personality` onto `ComposablePipeline` (Phase 5c).
- Extending `TTSBackend` Protocol with a `reset_per_session_state` method (Phase 5c).
- Tag plumbing / first-audio marker / `coroutine never awaited` warnings (Phase 5a sibling tasks per the exploration memo §4 — separate PRs).
