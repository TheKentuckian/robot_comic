# Spec — Telemetry housekeeping + startup-credentials triple-init guard

**Date:** 2026-05-16
**Status:** Implemented (single PR — `claude/telemetry-housekeeping-and-creds-guard`).
**Base commit:** `d26caf7` (end of Phase 4 epic, after PRs #383/#384/#385/#386
research memos merged).

## Background

The four research memos that closed the Phase 4 epic surfaced concrete bugs
and gaps that fall below the radar of a full follow-up phase but are too
real to leave unfixed:

- Instrumentation audit (PR #385) §3/§4 — three to four orphan OTel span
  attributes are `set_attribute()`'d on live spans but missing from
  `telemetry._SPAN_ATTRS_TO_KEEP`, so `CompactLineExporter` silently
  drops them on the way to the monitor.
- Instrumentation audit (PR #385) Rec 3/Rec 7 — two counters,
  `telemetry.errors` and `telemetry.playback_underruns`, are defined but
  have zero callers. They are wired here (Rec 3 / Rec 7), not retired.
- Boot memo (PR #383) §"weird things to know" #1 — all three composable
  adapters (LLM, TTS, STT) call `_prepare_startup_credentials()` on the
  shared host instance during their lifecycle. Idempotent today, but
  costs real wall-clock during the critical start_up path. Concretely,
  the Moonshine model load (~20 s on cold-boot) would otherwise be paid
  three times if any sub-step weren't already self-skipping; the httpx
  client constructor and Gemini client constructor *are* fully re-run on
  every call today.
- Boot memo (PR #383) §"weird things to know" #2 — the
  `handler.start_up.complete` supporting event fires *before*
  `pipeline.start_up()` is awaited. The row labelled "complete" is
  really "entered" — a naming bug that makes the monitor row
  semantically wrong.

## Scope (one PR)

### Fix 1 — Orphan OTel attribute allowlist (`telemetry._SPAN_ATTRS_TO_KEEP`)

Four attributes are now actually set on spans in `src/` (verified via grep
during spec authoring); all four are now in the allowlist so the exporter
preserves them:

| Attribute | Set at | Span |
|---|---|---|
| `gen_ai.usage.api_call_count` | `elevenlabs_tts.py:901` | `turn` |
| `gen_ai.server.time_to_first_token` | `base_realtime.py:1028`, `gemini_live.py:1010` | `llm.request` |
| `tts.voice_id` | `elevenlabs_tts.py:973` | `tts.synthesize` |
| `tts.char_count` | `elevenlabs_tts.py:974` | `tts.synthesize` |

All four were already set in production; this is purely an exporter
allowlist edit (zero new emission code). Verified with grep against `src/`
that each key has at least one set site before adding.

### Fix 2 — Dead counters wired

- **`telemetry.errors`** is now incremented at two Moonshine warning
  sites in `local_stt_realtime.py`:
  - `_MoonshineListener.on_error` — `inc_errors({"subsystem": "stt",
    "error_type": "stream_error"})`. The previously log-only
    `"Local STT error: %s"` warning now also lights the metrics surface.
  - `_log_heartbeat` idle-stall branch — `inc_errors({"subsystem": "stt",
    "error_type": "idle_stall"})`. This is the very symptom MOONSHINE_DIAG
    exists to debug (#314); the monitor should not have to scrape journald
    for it.
  - The third candidate site (`_rearm_local_stt_stream`) is intentionally
    left as `logger.debug` only — rearm is part of the documented #279
    workaround for normal operation, not an error condition. The
    instrumentation audit's "rearm as error" was the weakest recommendation
    and would generate ongoing noise on every utterance.
- **`telemetry.playback_underruns`** is now incremented in `warmup_audio.py`
  at both the threaded `_wait_and_emit_completion` and synchronous
  `_emit_completion_now` paths, whenever the welcome-WAV `aplay` Popen
  exits non-zero. The daemon owns the live ALSA sink so we can't watch
  TTS playback xruns from our process; the welcome-WAV exit code is the
  one playback-failure signal we own end-to-end. Rec 7.

No retire-the-counter path is taken: both counters have at least one
real, non-aspirational call site.

### Fix 3 — `_prepare_startup_credentials` idempotency guard

Added on `LocalSTTInputMixin._prepare_startup_credentials` — the
outer-most layer in MRO for the five `_LocalSTT*Host` classes that the
composable factory builds (see `handler_factory.py:131-159`). All three
adapter `prepare/start` paths funnel through this method on the shared
host:

- `LlamaLLMAdapter.prepare` → `handler._prepare_startup_credentials()`
- `GeminiLLMAdapter.prepare` → same
- `GeminiBundledLLMAdapter.prepare` → same
- `ElevenLabsTTSAdapter.prepare` → same
- `ChatterboxTTSAdapter.prepare` → same
- `GeminiTTSAdapter.prepare` → same
- `MoonshineSTTAdapter.start` → same (after dispatch swap)

Pattern: `self._startup_credentials_ready: bool`, checked on entry, set
to `True` only after the `super()._prepare_startup_credentials()` call
plus the Moonshine model load both succeed. **The flag is NOT set if any
step raises** — preserves the "don't lock out retries on failure"
semantics demanded by the spec.

Effect: the first adapter that calls in does the real work; the second
and third short-circuit cheaply.

### Fix 4 — `handler.start_up.complete` event timing

`ComposableConversationHandler.start_up` previously emitted
`handler.start_up.complete` *before* awaiting `self.pipeline.start_up()`.
This implementation moves the emit to a `try/finally` *after* the await
(Option A from the dispatch brief):

```python
try:
    await self.pipeline.start_up()
finally:
    # emit handler.start_up.complete with since_startup() ms
```

Rationale (recorded inline in the method docstring): "complete" should
mean handler-is-ready-to-accept-audio, which is true when
`pipeline.start_up()`'s `prepare/start` calls return — not when the
wrapper's own coroutine begins. The `try/finally` covers the
prepare-raised case so downstream monitor consumers see a row carrying
the failure point's elapsed time rather than no row at all (an "early
exit" signal instead of a hung "never arrived" state).

The existing test `test_start_up_emits_handler_start_up_complete_before_delegating`
was inverted to `..._after_delegating` and a new test
`test_start_up_emit_fires_even_when_pipeline_raises` was added to pin
the finally-branch behaviour. The previous emit-failure test (the
`telemetry.emit_supporting_event` raising path) was kept unchanged —
the `try/except` inside the finally block still swallows telemetry
exceptions.

## Out of scope

The following are explicitly *not* in this PR (they are Phase 5 5b's
sub-agent's lane or downstream follow-ups):

- `tool.execute` span on the composable orchestrator
  (`composable_pipeline.py:242`) — Phase 5 5b parallel sub-agent.
- `ComposablePipeline.tool_dispatcher` factory wiring — Phase 5 5b
  parallel sub-agent.
- Moonshine cold-load supporting event (`moonshine.model.loaded`) — Rec
  2 in instrumentation audit; warrants its own PR with a benchmark.
- Phase 5 follow-up: `pipeline.start_up.complete` *separate* event for
  pipeline-vs-handler readiness disambiguation. Today both still share
  the same row.

## Acceptance criteria (status)

- Allowlist now includes the 4 attributes. **Done** (Fix 1, `telemetry.py`).
- `telemetry.errors` has ≥1 real call site. **Done** (Fix 2,
  2 call sites in `local_stt_realtime.py`).
- `telemetry.playback_underruns` has ≥1 real call site. **Done** (Fix 2,
  2 call sites in `warmup_audio.py`).
- `_prepare_startup_credentials` is called only ONCE on cold boot.
  **Done** (Fix 3, guard on `LocalSTTInputMixin`). Verified via
  `test_prepare_startup_credentials_only_runs_once_for_shared_host`.
- `handler.start_up.complete` fires AFTER `pipeline.start_up()` returns.
  **Done** (Fix 4, `composable_conversation_handler.py`). Verified via
  `test_start_up_emits_handler_start_up_complete_after_delegating`.
- All existing tests still pass. (See PR description for the green run.)
- `uvx ruff@0.12.0 check` + `format --check` green from repo root.
- `mypy --pretty <changed files>` green.
- `pytest tests/ -q` green.

## Phase 5 follow-up — link to PR #374 spec

PR #374's spec at `docs/superpowers/specs/2026-05-16-lifecycle-boot-timeline-events.md`
should be amended in a follow-up to reflect that the
`handler.start_up.complete` event timing was corrected here (Fix 4
above). The legacy emit site in `elevenlabs_tts.py:362` is unchanged in
this PR — the legacy realtime handler emits at the same point the
composable wrapper used to (i.e. *before* its long-running work), and
fixing that is a separate cleanup once the composable factory is the
default path. The two emit sites are presently asymmetric; calling it
out so the monitor's row interpretation can be made site-aware later.
