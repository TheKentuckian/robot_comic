# Lifecycle Hook #3 — Boot-timeline supporting events on the composable path

**Branch:** `claude/lifecycle-boot-timeline-events`
**Epic:** #337 (pipeline refactor) — Deferred lifecycle hooks
**Issue:** #321 (boot-timeline supporting events) / #301 (monitor TUI)
**Date:** 2026-05-16
**Author:** sub-agent (manager-driven)

## Background — which boot-timeline events does the composable path drop?

`PIPELINE_REFACTOR.md` line 313 lists boot-timeline supporting events from
#321 as a deferred lifecycle hook with the new home:

> `ComposableConversationHandler.start_up()` before delegating to pipeline

PR #321 introduced four boot-timeline supporting rows on the monitor:

| Event | Emission site | Surface |
|-------|---------------|---------|
| `app.startup` | `main.py:232` | App boot, not handler-bound. |
| `welcome.wav.played` | `main.py:82` (early dispatch) + `warmup_audio.py:480` (in-process) | Boot, not handler-bound. |
| `handler.start_up.complete` | `ElevenLabsTTSResponseHandler.start_up` (`elevenlabs_tts.py:364`) | Per-handler. |
| `first_greeting.tts_first_audio` | TTS modules' frame-enqueue sites (`elevenlabs_tts.py:1000`, `chatterbox_tts.py:370`, `gemini_tts.py:416`, `gemini_live.py:1036`, `llama_gemini_tts.py:172`, `base_realtime.py:1034`) | Per-handler. |

Investigation results for the composable path (`FACTORY_PATH=composable`):

### `app.startup` — preserved

Fires from `main.py:232`, the same path both legacy and composable boot through.
No fix needed.

### `welcome.wav.played` — preserved

Fires from `main.py:82` (early-welcome dispatch) before any handler is built,
and from `warmup_audio.py:480` (in-process fallback). Both paths run regardless
of factory choice. No fix needed.

### `handler.start_up.complete` — NOT preserved

`ElevenLabsTTSResponseHandler.start_up` emits this directly. On the composable
path, `LocalStream` awaits `_handler.start_up()` where `_handler` is a
`ComposableConversationHandler`. That wrapper's `start_up` delegates to
`self.pipeline.start_up()` (`ComposablePipeline.start_up`), which never
invokes the legacy handler's `start_up`. The legacy ElevenLabs handler held by
reference as `self._tts_handler` is constructed but its `start_up` is never
awaited.

`composable_conversation_handler.py:103` even carries a TODO making this
explicit:

```python
async def start_up(self) -> None:
    """Delegate to :meth:`ComposablePipeline.start_up` — blocks until shutdown."""
    # TODO(phase4-lifecycle): emit the four boot-timeline supporting events
    # from #321 before delegating; composable mode currently drops them.
    await self.pipeline.start_up()
```

**Impact at 4d default flip:** monitor boot-timeline lane misses the
`handler.start_up.complete` row, so operators can no longer read the
boot-to-handler-ready window directly off the TUI.

### `first_greeting.tts_first_audio` — mixed; out of scope for this PR

| Adapter | Delegates to | Emission? |
|---------|--------------|-----------|
| `ElevenLabsTTSAdapter.synthesize` | `_stream_tts_to_queue` (elevenlabs_tts.py:1000) | **Preserved** — site reached. |
| `ChatterboxTTSAdapter.synthesize` | `_synthesize_and_enqueue` (chatterbox_tts.py:370) | **Preserved** — site reached. |
| `GeminiTTSAdapter.synthesize` | Inlines its own per-sentence loop; does NOT call legacy `_dispatch_completed_transcript` (gemini_tts.py:416) | **Dropped** for bundled-Gemini composable triples. |

The Gemini-TTS adapter gap is real but is a *TTS adapter* fix, not a
handler-`start_up` fix. The doc-table entry for this hook scopes the new home
to `ComposableConversationHandler.start_up()` only. Fixing the GeminiTTSAdapter
emission is filed as a follow-up so the blast radius stays minimal and the
adapter test surface (`tests/adapters/test_gemini_tts_adapter.py`) is touched
in its own PR. See "Out of scope" below.

## Decision — fix shape

Emit `handler.start_up.complete` from `ComposableConversationHandler.start_up()`
before delegating to `self.pipeline.start_up()`. Mirror the legacy emission's
shape exactly:

```python
from robot_comic.startup_timer import since_startup
from robot_comic import telemetry as _telemetry

try:
    _telemetry.emit_supporting_event(
        "handler.start_up.complete",
        dur_ms=since_startup() * 1000,
    )
except Exception:
    # Telemetry must never break boot.
    pass
await self.pipeline.start_up()
```

The emit fires *before* the pipeline's `await self._stop_event.wait()` so the
supporting-event row lands on the monitor at "handler ready", not at
"handler shutdown" (the same bug PR #337's elevenlabs fix already addressed
on the legacy side).

The `try/except` and lazy imports match the legacy pattern at
`elevenlabs_tts.py:360–369` byte-for-byte (modulo the import paths). Telemetry
must never block boot, and the `since_startup()` import is deferred so the
hot path of `start_up` doesn't pay it on cold boot when telemetry is disabled.

Idempotency: `emit_supporting_event` is fire-and-forget; the wrapper's
`start_up` is called exactly once per `LocalStream` lifecycle, so a once-guard
is not needed. The legacy ElevenLabs path uses the same fire-and-forget
shape.

## Scope

| File | Change |
|------|--------|
| `src/robot_comic/composable_conversation_handler.py` | Add `handler.start_up.complete` emission to `start_up()` immediately before delegating; remove the TODO comment that now describes the implemented behaviour. |
| `tests/test_composable_conversation_handler.py` | Add two regression tests: (a) the emit fires before `pipeline.start_up()` is awaited; (b) the emit failing does not break `start_up()`. |

## Files NOT touched

- `src/robot_comic/elevenlabs_tts.py` — legacy emission stays; covers the
  legacy-path mode that's still production until Phase 4d.
- `src/robot_comic/composable_pipeline.py` — telemetry stays on the handler
  surface, not the orchestrator. Reason: the orchestrator is used by all
  composable triples, including future ones; routing through the handler
  wrapper keeps the emission tied to "the handler was activated", which is
  the semantic the legacy emit captures.
- `src/robot_comic/adapters/*.py` — `first_greeting.tts_first_audio` on the
  GeminiTTSAdapter is a separate (follow-up) PR.
- `src/robot_comic/console.py` — wrapper site (line 1209) already removed the
  emit per PR #337 commentary; no change here.
- `src/robot_comic/telemetry.py`, `src/robot_comic/startup_timer.py` — APIs
  used as-is.
- `ConversationHandler` ABC, `handler_factory.py` — no surface changes.

## Why the emit lives in `ComposableConversationHandler`, not
`ComposablePipeline`

Three reasons:

1. **Semantics.** `handler.start_up.complete` is conceptually "the handler
   became ready to accept user audio". The composable handler wrapper is the
   `ConversationHandler` instance the rest of the app holds — emitting on its
   `start_up` matches the row's meaning. The pipeline is an internal
   implementation detail.
2. **Symmetry with the legacy fix.** PR #337's elevenlabs fix moved the emit
   *into the handler that becomes ready*. Doing the same on the composable
   side keeps the two paths symmetric — one emission per
   `ConversationHandler.start_up`, regardless of which path.
3. **Doc-table fidelity.** `PIPELINE_REFACTOR.md` literally names
   `ComposableConversationHandler.start_up()` as the new home.

## Acceptance criteria

- Two new regression tests in `tests/test_composable_conversation_handler.py`:
  - `test_start_up_emits_handler_start_up_complete_before_delegating` — asserts
    the emit fires (with a numeric `dur_ms`) *before* `pipeline.start_up` is
    awaited.
  - `test_start_up_emit_failure_does_not_break_pipeline_delegation` — asserts
    that a raised `emit_supporting_event` is swallowed and `pipeline.start_up`
    is still awaited.
- All existing tests still pass.
- `uvx ruff@0.12.0 check` and `format --check` green from repo root.
- `.venv/Scripts/mypy --pretty src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py` green.
- `.venv/Scripts/pytest tests/ -q` green.
- The TODO at `composable_conversation_handler.py:103` is removed, replaced
  with a brief docstring note about the emit.

## Out of scope

- **`GeminiTTSAdapter.synthesize` missing `emit_first_greeting_audio_once`.**
  This is the only composable TTS surface where the legacy emit was dropped
  (the other two delegate cleanly). It's a small TTS-adapter fix that
  deserves its own PR with adapter test surface. Filed as a follow-up.
- **Removing the legacy emit site in `elevenlabs_tts.py:360–369`.** That's
  Phase 4e legacy retirement, after 4d default flip.
- **Adding `app.startup` / `welcome.wav.played` emits to the wrapper.** Both
  fire from `main.py` already on both paths; emitting from the wrapper would
  double them.
- **Reconciling other legacy handlers that never emitted
  `handler.start_up.complete`** (`GeminiTTSResponseHandler`, `BaseLlamaResponseHandler`,
  `GeminiLiveHandler`). Those are pre-existing legacy gaps. The composable
  emit closes the gap *for all composable triples uniformly*, which is a
  small improvement consistent with Phase 4c.5's GeminiBundledLLMAdapter
  telemetry stance (one consistent surface on the new path).
