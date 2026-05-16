# Phase 5e.3 — Migrate `(moonshine, llama, chatterbox)` off `LocalSTTInputMixin`

**Date:** 2026-05-16
**Status:** Spec — implementation on
`claude/phase-5e-3-llama-chatterbox-decouple`.
**Tracks:** epic #391; established pattern from
`docs/superpowers/specs/2026-05-16-phase-5e-2-llama-elevenlabs-decouple.md`
§4.
**Predecessor:** Phase 5e.2 (`#407`, migrated `(moonshine, llama,
elevenlabs)` off the mixin + locked the pattern).

---

## §1 — Scope

Mechanical mirror of 5e.2 for the `(moonshine, llama, chatterbox)`
triple. **No new design decisions.** The Protocol, `ComposablePipeline`
host-concern landing, `MoonshineSTTAdapter` standalone wiring, and
`_clear_queue` double-mirror in `ComposableConversationHandler` are
all already in place from 5e.2 and unchanged here.

This PR:

1. Adds the per-handler `_startup_credentials_ready` idempotency guard
   to `ChatterboxTTSResponseHandler._prepare_startup_credentials` so the
   dual-adapter `prepare` calls don't re-run health probes / KV-cache
   warmup / TTS warmup / voice-clone reference loading. Without the
   guard each `prepare` triggers a fresh `_warmup_llm_kv_cache` (a
   streaming POST to llama-server) and `_warmup_tts` (a TTS round-trip
   to Chatterbox) — load-bearing.
2. Rewrites `_build_composable_llama_chatterbox` to construct a plain
   `ChatterboxTTSResponseHandler` (no mixin shell), a standalone
   `MoonshineSTTAdapter(should_drop_frame=...)`, and passes `deps` +
   `welcome_gate` into `ComposablePipeline` — identical shape to 5e.2's
   `_build_composable_llama_elevenlabs`.
3. Deletes `_LocalSTTLlamaChatterboxHost` (only call site is the
   builder we're rewriting).
4. Adds factory tests mirroring 5e.2: standalone STT, `should_drop_frame`
   wired, `pipeline.deps is mock_deps`.

The other three remaining triples (`gemini_chatterbox`,
`gemini_elevenlabs`, `gemini_tts`) keep their `_LocalSTT*Host` mixin
hosts unchanged. 5e.4–5e.6 retire them one at a time.

## §2 — Chatterbox-specific notes

### §2.1 — `_prepare_startup_credentials` guard

Today's body
(`src/robot_comic/chatterbox_tts.py:125-144`):

```python
async def _prepare_startup_credentials(self) -> None:
    await super()._prepare_startup_credentials()
    profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    if profile:
        self._voice_clone_ref_path = load_voice_clone_ref(config.PROFILES_DIRECTORY / profile)
    else:
        self._voice_clone_ref_path = None
    logger.info(...)
    await self._probe_llama_health()
    await self._warmup_llm_kv_cache()
    await self._warmup_tts()
```

Post-5e.3:

```python
async def _prepare_startup_credentials(self) -> None:
    if getattr(self, "_startup_credentials_ready", False):
        return
    await super()._prepare_startup_credentials()
    # ... existing body unchanged ...
    self._startup_credentials_ready = True
```

The flag is set **after** all three warmup calls return (the same shape
5e.2 used on `LlamaElevenLabsTTSResponseHandler`). A mid-warmup
exception leaves the flag false so a retry will re-attempt the full
chain. This matches the 5e.2 contract verbatim.

### §2.2 — Factory builder rewrite

Identical shape to `_build_composable_llama_elevenlabs`. The
`ChatterboxTTSAdapter` Protocol surface accepts a plain
`ChatterboxTTSResponseHandler` (today's builder already passes the
mixin-shelled `_LocalSTTLlamaChatterboxHost` which `isinstance`-
satisfies the same surface because the host subclasses the handler).
Echo-guard closure reads `host._speaking_until` set by
`_enqueue_audio_frame` in `BaseLlamaResponseHandler` (same as
ElevenLabs — both inherit from the same base).

### §2.3 — Out of scope

- Other triples (5e.4-5e.6).
- Touching `LocalSTTInputMixin`, `ComposablePipeline`,
  `MoonshineSTTAdapter`, `STTBackend` Protocol, or `BaseRealtimeHandler`.
- The `_clear_queue` double-mirror (5e.6 cleanup).
- Heartbeat / `MOONSHINE_DIAG` instrumentation (still punted, per 5e.1).

## §3 — Tests

New / updated assertions in `tests/test_handler_factory.py`:

- `test_moonshine_llama_chatterbox_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_llama_chatterbox_wires_should_drop_frame_callback` —
  `result.pipeline.stt._should_drop_frame is not None` and returns
  `False` at default (`_speaking_until` is 0.0).
- `test_moonshine_llama_chatterbox_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_moonshine_chatterbox_default_llama_routes_to_composable`
keeps passing (asserts `isinstance(_tts_handler,
ChatterboxTTSResponseHandler)`; the plain handler is what the wrapper
now wraps).

New `tests/test_chatterbox_tts.py`:

- `test_prepare_startup_credentials_is_idempotent` — second call must
  NOT re-invoke `super()._prepare_startup_credentials` /
  `_warmup_llm_kv_cache` / `_warmup_tts` / `_probe_llama_health`.

## §4 — Verification

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/chatterbox_tts.py \
    src/robot_comic/handler_factory.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Known flakes (do NOT fix; re-run if hit):

- `test_huggingface_realtime::test_run_realtime_session_passes_allocated_session_query`
- `test_openai_realtime::test_openai_excludes_head_tracking_when_no_head_tracker`
- `test_handler_factory::test_moonshine_openai_realtime_output`,
  `test_moonshine_hf_output`
- `test_gemini_turn_buffers_transcripts_and_schedules_motion_reset`

## §5 — Risk

Low. The pattern was proven by 5e.2; the chatterbox handler shares the
`BaseLlamaResponseHandler` ancestry, the same `_speaking_until`
semantics, and the same dual-adapter `prepare` call shape. The only
chatterbox-specific concern is the guard's placement around the three
warmup calls — `_warmup_llm_kv_cache` and `_warmup_tts` are
fire-and-forget (try/except → WARNING + return), so a partial failure
mid-`_prepare_startup_credentials` won't trip the guard. We set the
flag only on the success path; failed runs re-attempt the full chain
on a retried `prepare`.

## §6 — Diff budget

Target: ≤300 LOC across all touched files (well under the 600 LOC
trip-wire from the brief).

- `chatterbox_tts.py`: +4 LOC (guard).
- `handler_factory.py`: -3 LOC (class delete + builder rewrite ~net
  parity).
- `tests/test_handler_factory.py`: +45 LOC (three new tests).
- `tests/test_chatterbox_tts.py` (new): +60 LOC (idempotency test +
  helpers).

Total: ~110 LOC. Pattern is mechanical; if it grows past 300 LOC,
stop and report.
