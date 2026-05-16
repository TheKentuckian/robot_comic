# TDD Plan — Telemetry housekeeping + startup-credentials triple-init guard

Companion to `docs/superpowers/specs/2026-05-16-telemetry-housekeeping-and-creds-guard.md`.

One branch (`claude/telemetry-housekeeping-and-creds-guard`), four
fix-shaped commits, one cleanup commit if needed.

## Commits

### Commit 1 — Fix 1: allowlist exposure (`telemetry._SPAN_ATTRS_TO_KEEP`)

- [x] Grep `src/` for each of the 4 candidate attribute names; confirm
      at least one set site per attribute.
- [x] Add `gen_ai.usage.api_call_count`, `gen_ai.server.time_to_first_token`,
      `tts.voice_id`, `tts.char_count` to `_SPAN_ATTRS_TO_KEEP` with a
      comment naming the audit recommendation.
- [x] Test (RED → GREEN): `test_orphan_span_attributes_now_in_allowlist`
      parameterised over the four keys.
- [x] Test: `test_compact_line_exporter_keeps_new_attributes` — build a
      fake span with all four attributes plus one non-allowlisted, run
      the exporter, confirm the four survive and the non-allowlisted one
      is filtered.

### Commit 2 — Fix 2a: wire `telemetry.errors` at Moonshine warning sites

- [x] Locate `_MoonshineListener.on_error` and the `_log_heartbeat`
      idle-stall warning in `local_stt_realtime.py`.
- [x] Add `telemetry.inc_errors({"subsystem": "stt",
      "error_type": "stream_error" | "idle_stall"})` at each.
- [x] Test (RED → GREEN): `test_inc_errors_called_from_local_stt_on_error_listener`
      builds a `_MoonshineListener` shell, patches `telemetry.inc_errors`,
      drives `on_error`, asserts a call with the right attribute set.
- [x] Test: `test_inc_errors_is_safe_when_uninitialised` — guard against
      a regression where the helper assumes the counter is built.
- [x] Test: `test_inc_errors_arg_shape_matches_call_sites` — pins the
      `(attrs)` signature so future call sites don't pass count
      positionally.

### Commit 3 — Fix 2b: wire `telemetry.playback_underruns` at welcome-WAV completion

- [x] Locate `_wait_and_emit_completion` (threaded path) and
      `_emit_completion_now` (sync path) in `warmup_audio.py`.
- [x] Add `_telemetry.inc_playback_underruns({"path": "welcome.wav"})`
      after each `emit_supporting_event(..., aplay.exit_code=...)` call
      gated on `exit_code != 0` (and `is not None` on the sync path).
- [x] Test (RED → GREEN): `test_inc_playback_underruns_fires_on_nonzero_aplay_exit`
      patches `inc_playback_underruns` + `emit_supporting_event`, runs
      `_emit_completion_now` with a `returncode=1` Popen, asserts one call.
- [x] Test: `test_inc_playback_underruns_skipped_on_clean_exit` — zero
      exit must not fire.
- [x] Test: `test_inc_playback_underruns_is_safe_when_uninitialised`.

### Commit 4 — Fix 3: idempotency guard on `_prepare_startup_credentials`

- [x] Locate `LocalSTTInputMixin._prepare_startup_credentials` (the
      outermost-in-MRO method for every composable host).
- [x] Add `self._startup_credentials_ready: bool` attribute; check at
      entry, set `True` only after the full body succeeds. (Failure must
      leave it `False` so retries re-attempt.)
- [x] Test (RED → GREEN): `test_prepare_startup_credentials_only_runs_once_for_shared_host`
      builds a `_Host(LocalSTTInputMixin, _CountingBase)` and calls
      the method three times; asserts the counting-base super call only
      fired once.
- [x] Test: `test_prepare_startup_credentials_retries_after_failure`
      pins the retry-on-failure semantics — first call raises, flag stays
      False, second call succeeds, third call short-circuits.

### Commit 5 — Fix 4: move `handler.start_up.complete` to after `pipeline.start_up`

- [x] Rewrite `ComposableConversationHandler.start_up` to await the
      pipeline first, then emit in a `try/finally`.
- [x] Update docstring with the rationale (boot memo + audit citations).
- [x] Replace existing test
      `test_start_up_emits_handler_start_up_complete_before_delegating`
      with the inverted assertion
      (`..._after_delegating`, `pipeline_seen_emits == [0]`).
- [x] Add `test_start_up_emit_fires_even_when_pipeline_raises` to pin
      the `try/finally` branch.
- [x] Leave `test_start_up_emit_failure_does_not_break_pipeline_delegation`
      unchanged — the inner `try/except` semantics survive.

### Commit 6 — Spec + plan (this file)

- [x] `docs/superpowers/specs/2026-05-16-telemetry-housekeeping-and-creds-guard.md`.
- [x] `docs/superpowers/plans/2026-05-16-telemetry-housekeeping-and-creds-guard.md`.

## Verification

- `uvx ruff@0.12.0 check` + `format --check` from repo root.
- `mypy --pretty` over the changed files.
- `pytest tests/ -q` (with the documented local-env quirks ignored if
  they bite).
