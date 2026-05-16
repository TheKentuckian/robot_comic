# Lifecycle Hook #3b — `GeminiTTSAdapter.first_greeting.tts_first_audio` emission

**Branch:** `claude/lifecycle-hook-3b-gemini-tts-first-greeting`
**Epic:** #337 (pipeline refactor) — Deferred lifecycle hooks
**Issue:** #321 (boot-timeline supporting events) / #301 (monitor TUI)
**Predecessor:** PR #374 (Hook #3, `handler.start_up.complete` on the composable
path). That PR's spec
(`docs/superpowers/specs/2026-05-16-lifecycle-boot-timeline-events.md`,
"Out of scope") explicitly flagged this as a follow-up.
**Date:** 2026-05-16

## Background — what's the gap?

`PIPELINE_REFACTOR.md`'s deferred-lifecycle-hooks table lists four
boot-timeline supporting events from #321. Three are already preserved on the
composable path; the fourth — `first_greeting.tts_first_audio` — is preserved
for two of the three composable TTS adapters but **dropped** by
`GeminiTTSAdapter.synthesize`.

| Composable TTS adapter | Delegates to | `first_greeting.tts_first_audio` emit |
|------------------------|--------------|----------------------------------------|
| `ElevenLabsTTSAdapter.synthesize` | `_stream_tts_to_queue` (`elevenlabs_tts.py:999`) | Preserved — legacy emits inside the first-PCM branch. |
| `ChatterboxTTSAdapter.synthesize` | `_synthesize_and_enqueue` (`chatterbox_tts.py:368`) | Preserved — legacy emits per frame; helper is once-per-process. |
| `GeminiTTSAdapter.synthesize` | Re-implements the per-sentence loop inline; never reaches legacy `_dispatch_completed_transcript` (`gemini_tts.py:415`) | **Dropped.** |

Verification:

- `src/robot_comic/telemetry.py:347-368` — `emit_first_greeting_audio_once`
  is the canonical entry point and is once-per-process (`_FIRST_GREETING_EMITTED`
  flag).
- `src/robot_comic/adapters/gemini_tts_adapter.py:142-188` — the adapter's
  `synthesize` chunks `_call_tts_with_retry`'s PCM bytes directly and yields
  `AudioFrame`s without going through the legacy `_dispatch_completed_transcript`
  call that owns the emit.

Net: on the composable triple `(moonshine, gemini_tts)`, the monitor's
boot-timeline lane never sees the `first_greeting.tts_first_audio` row.

## Decision — fix shape

Add a call to `telemetry.emit_first_greeting_audio_once()` inside
`GeminiTTSAdapter.synthesize`, fired on the **first PCM frame the adapter
yields** for any given session. Because the helper has a process-level
once-guard, calling it on every yielded frame is safe and matches the legacy
behaviour exactly — but for cleanliness we mirror the legacy
`gemini_tts.py:415` placement: a per-frame call inside the inner
`_pcm_to_frames` loop, immediately before yielding. The helper short-circuits
all subsequent calls.

This matches `ChatterboxTTSAdapter`'s delegated emission shape — every frame
yields the call; the helper deduplicates.

Why per-frame and not "first-call-only via local flag":

1. **Matches legacy.** `gemini_tts.py:415` calls
   `emit_first_greeting_audio_once()` inside the inner `_pcm_to_frames`
   loop, once per frame. The helper's process-level flag is the canonical
   dedupe mechanism.
2. **No new per-adapter state.** Avoids a new instance flag on the adapter
   class.
3. **Symmetric with the other adapters.** `_stream_tts_to_queue` (the
   ElevenLabs path) emits *once* (gated on its `got_audio` local flag), but
   `chatterbox_tts.py:368` emits per frame; the helper deduplicates either
   way. Per-frame is the simpler shape.

## Scope

| File | Change |
|------|--------|
| `src/robot_comic/adapters/gemini_tts_adapter.py` | Add `telemetry.emit_first_greeting_audio_once()` inside the per-frame yield loop in `synthesize`. |
| `tests/adapters/test_gemini_tts_adapter.py` | New regression test: after a `synthesize` that yields ≥1 frame, `emit_first_greeting_audio_once` was called at least once. |
| `PIPELINE_REFACTOR.md` | Replace the 4f row's `#TBD (manager fixes on merge)` with `#381 (commit 8873fa2)`. |

## Files NOT touched

- `src/robot_comic/gemini_tts.py` — legacy emit at line 415 stays.
- `src/robot_comic/telemetry.py` — `emit_first_greeting_audio_once` is the
  pre-existing API; no signature change.
- `src/robot_comic/composable_pipeline.py` / `composable_conversation_handler.py`
  — telemetry belongs on the adapter (the TTS frame-enqueue surface), not the
  orchestrator. Reason: the orchestrator runs the same code path for all three
  TTS adapters; the emission is adapter-specific because two of the three
  preserve it via delegation and only the Gemini-TTS adapter needs a direct
  call.

## Why the emit lives in `GeminiTTSAdapter`, not the pipeline

Three reasons:

1. **Semantics.** `first_greeting.tts_first_audio` is conceptually "the first
   PCM frame of the boot turn left this TTS surface". The adapter is that
   surface; the pipeline is a downstream consumer.
2. **Symmetry with the other adapters.** ElevenLabs and Chatterbox preserve
   the emit through delegation to legacy TTS methods that own the call. The
   Gemini-TTS adapter doesn't delegate (it inlines), so the emit has to live
   inline.
3. **Doc-table fidelity.** `PIPELINE_REFACTOR.md`'s deferred-lifecycle-hooks
   row for `first_greeting.tts_first_audio` lists "TTS modules' frame-enqueue
   sites" as the home; the adapter's per-frame yield loop is the structural
   analogue on the composable path.

## Does the legacy `GeminiTTSResponseHandler` emit this event?

**Yes** — at `gemini_tts.py:415`, inside the per-frame loop in
`_dispatch_completed_transcript`. The legacy bundled-Gemini path is fine; only
the composable path through `GeminiTTSAdapter.synthesize` drops it.

This makes the fix purely a parity restoration, not a new emission.

## Acceptance criteria

- New regression test in `tests/adapters/test_gemini_tts_adapter.py`:
  - `test_synthesize_emits_first_greeting_tts_first_audio_on_first_frame` —
    monkeypatches `robot_comic.telemetry.emit_first_greeting_audio_once`,
    runs `synthesize("Hello.")` through a stub handler returning a single PCM
    chunk, asserts the helper was called at least once.
- All existing tests still pass.
- `uvx ruff@0.12.0 check` and `format --check` green from repo root.
- `.venv/Scripts/mypy --pretty src/robot_comic/adapters/gemini_tts_adapter.py
  tests/adapters/test_gemini_tts_adapter.py` green.
- `.venv/Scripts/pytest tests/ -q` green.
- `PIPELINE_REFACTOR.md` line 24 reads `#381 (commit 8873fa2)`.

## Out of scope

- **Plumbing `first_audio_marker` through `TTSBackend`.** The
  `ElevenLabsTTSAdapter` already drops the marker (see its module docstring
  "Known gap"); restoring it requires a Protocol-level metadata channel and
  is its own follow-up. Bringing back the helper-level emit is enough to
  re-light the boot-timeline row; the `robot.tts.time_to_first_audio`
  histogram on the composable path remains unrecorded until the marker
  plumbing lands.
- **`gen_ai.system` attribute on the emit.** Today's
  `emit_first_greeting_audio_once` takes no attributes; matching shape across
  adapters.
- **Legacy retirement.** `gemini_tts.py:415`'s emit stays — Phase 4e+ removes
  legacy handlers wholesale.
