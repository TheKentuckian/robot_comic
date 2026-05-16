# Phase 4f implementation plan — Retire BACKEND_PROVIDER + LOCAL_STT_RESPONSE_BACKEND

Spec: `docs/superpowers/specs/2026-05-16-phase-4f-backend-provider-retirement.md`.

This is a single PR (~600-1000 LOC, 5-6 commits) on branch
`claude/phase-4f-backend-provider-retirement`.

## Commit sequence

### Commit 1 — Surface 1: `prompts.py` delivery-tags gate

- Update `_uses_gemini_tts` to take a single `output_backend` argument and
  return `output_backend == AUDIO_OUTPUT_GEMINI_TTS`.
- Update `_filter_delivery_tags` to read `config.AUDIO_OUTPUT_BACKEND`.
- Adapt `tests/test_prompt_backend_awareness.py` to construct a fake config
  with `AUDIO_OUTPUT_BACKEND` (not `BACKEND_PROVIDER` /
  `LOCAL_STT_RESPONSE_BACKEND`). Parametric test cases mirror the legacy
  table: the two `*_gemini_tts` outputs keep the section, everything else
  strips.
- Update `tests/test_prompt_live_styling.py` likewise.

### Commit 2 — Surface 2: rename ClassVar to PROVIDER_ID and rename voice helpers

- Rename `BACKEND_PROVIDER: ClassVar[str]` to `PROVIDER_ID: ClassVar[str]`
  on `BaseRealtimeHandler`. Update the `_REQUIRED_PROVIDER_CONFIG` tuple
  and the `__init_subclass__` error message.
- Update `HuggingFaceRealtimeHandler`, `OpenaiRealtimeHandler`,
  `LocalSTTOpenAIRealtimeHandler`, `LocalSTTHuggingFaceRealtimeHandler` to
  set `PROVIDER_ID` instead of `BACKEND_PROVIDER`.
- Update the 3 voice-catalog call sites in `base_realtime.py` +
  `openai_realtime.py` to pass `self.PROVIDER_ID`.
- Update the 7 OTel attribute emission sites in `base_realtime.py` to use
  `self.PROVIDER_ID`. *The emitted attribute string values do not change.*
- Rename `get_available_voices_for_backend` →
  `get_available_voices_for_provider`, `get_default_voice_for_backend` →
  `get_default_voice_for_provider` in `config.py`. Update all call sites
  (`prompts.py` get_session_voice fallback, `local_stt_realtime.py`
  `LocalSTTOpenAIRealtimeHandler.get_current_voice`).
- Update `tests/test_huggingface_realtime.py`, `tests/test_openai_realtime.py`
  to monkeypatch `PROVIDER_ID` instead of `config.BACKEND_PROVIDER` where
  the value drives a code path (or just stop monkeypatching when it's
  unused).

### Commit 3 — Surface 5 (partial): `config.py` introduces new helpers, keeps dials

This is a transitional commit — the old dial values still get set on the
Config class so Surface 3's POST handler keeps working until Commit 4
finishes the admin UI rewrite. The goal is to introduce the new helpers
that Surface 3 needs:

- Add `provider_id_from_pipeline(pipeline_mode, audio_output_backend) -> str`
  that returns one of `OPENAI_BACKEND`, `GEMINI_BACKEND`, `HF_BACKEND`,
  `LOCAL_STT_BACKEND`. Used by `_status_payload`'s `active_backend` field
  and by helper functions that need to look up which API key is needed for
  the current pipeline.
- Add `get_provider_id() -> str` (renames `get_backend_choice()` but
  without the legacy `model_name` fallback — provider derives purely from
  PIPELINE_MODE + AUDIO_OUTPUT_BACKEND).
- Update `get_backend_label` to compute from `PIPELINE_MODE` +
  `AUDIO_OUTPUT_BACKEND` instead of the dial.

### Commit 4 — Surface 3: admin UI route + persist logic

- Replace `BackendPayload` Pydantic model: drop `backend` and
  `local_stt_response_backend` fields; add `pipeline_mode`,
  `audio_input_backend`, `audio_output_backend`. Validate
  `pipeline_mode in PIPELINE_MODE_CHOICES` and that
  `audio_input_backend`/`audio_output_backend` are valid when
  `pipeline_mode == PIPELINE_MODE_COMPOSABLE`.
- Replace `_persist_backend_choice(backend)` with
  `_persist_pipeline_choice(pipeline_mode, audio_input, audio_output)`.
  Writes the three new env vars. Drops the `BACKEND_PROVIDER` / `MODEL_NAME`
  writes.
- Replace `_active_backend_name = get_backend_choice()` initialisation
  with `_active_provider_id = get_provider_id()`. `_active_backend()` then
  returns the provider_id directly.
- Update `_status_payload` to emit the new field set (remove
  `backend_provider`, `local_stt_response_backend`,
  `local_stt_response_backend_choices`; add `pipeline_mode` +
  `audio_input_backend` + `audio_output_backend` + their `_choices` lists).
- Update credential-gating helpers (`_has_required_key`, `_requirement_name`)
  to read `config.AUDIO_OUTPUT_BACKEND` instead of
  `config.LOCAL_STT_RESPONSE_BACKEND`.
- Update the POST handler logic to dispatch on
  `(pipeline_mode, audio_output_backend)` instead of
  `(backend, local_stt_response_backend)`.
- Update `tests/test_console.py` (~20 hits):
  - Replace `monkeypatch.setattr(config, "BACKEND_PROVIDER", ...)` with
    `monkeypatch.setattr(config, "PIPELINE_MODE", ...)` plus
    `AUDIO_OUTPUT_BACKEND`/`AUDIO_INPUT_BACKEND` where relevant.
  - Replace POST request bodies: `{"backend": "..."}` becomes
    `{"pipeline_mode": "..."}` (plus `audio_*_backend` fields).
  - Replace assertions on `data["backend_provider"]` with
    `data["pipeline_mode"]` and / or `data["active_backend"]`.
  - Replace assertions on `.env` content: `BACKEND_PROVIDER=...` becomes
    `REACHY_MINI_PIPELINE_MODE=...` etc.
- Update `tests/test_admin_pipeline_3column.py` to match new POST shape.

### Commit 5 — Surface 4: frontend JSON wiring

- `static/main.js`:
  - Map `status.pipeline_mode` + `status.audio_output_backend` → family
    radio (back-compat read).
  - On save, build `pipeline_mode` + `audio_input_backend` +
    `audio_output_backend` from selected family + 3-column choices.
  - Drop reads of `status.backend_provider` and
    `status.local_stt_response_backend` and writes of `body.backend` /
    `body.local_stt_response_backend`.
- `static/index.html`: no markup change required (the radios are
  presentational only).

### Commit 6 — Surface 5 (rest) + cleanup

- Delete dial-specific code in `config.py`:
  - Remove `BACKEND_PROVIDER`, `LOCAL_STT_RESPONSE_BACKEND` class
    attributes.
  - Remove `_normalize_backend_provider`, `_normalize_local_stt_response_backend`,
    `_resolve_model_name`, `DEFAULT_BACKEND_PROVIDER`.
  - Remove `LOCAL_STT_RESPONSE_BACKEND_ENV`,
    `LOCAL_STT_RESPONSE_BACKEND_CHOICES`,
    `LOCAL_STT_DEFAULT_RESPONSE_BACKEND`, `_LOCAL_STT_RESPONSE_TO_AUDIO_OUTPUT`.
  - Adapt `derive_audio_backends` to take a `provider_id` argument (still
    useful for testing and the `audio-backends.md` matrix) — but no longer
    used by the live config-load path.
  - Replace the BACKEND_PROVIDER reads in `_resolved_audio` /
    `refresh_runtime_config_from_env` with a direct env-var read of
    `REACHY_MINI_AUDIO_INPUT_BACKEND` / `REACHY_MINI_AUDIO_OUTPUT_BACKEND`,
    with fallback derivation from `REACHY_MINI_PIPELINE_MODE`.
  - Leave a one-line retirement-explainer comment.
- Update `main.py:240-251` to log a single `logger.info("Configured pipeline mode: %s (%s)", config.PIPELINE_MODE, get_backend_label())` line.
- Update `.env.example` to drop `BACKEND_PROVIDER` and
  `LOCAL_STT_RESPONSE_BACKEND` rows, add `REACHY_MINI_PIPELINE_MODE`
  pointer.
- Update `tests/test_config_name_collisions.py` and other test files that
  reference the old dial.
- Update `tests/integration/test_handler_factory_smoke.py` for the
  `derive_audio_backends(provider_id, ...)` rename if it stays around, or
  rewrite for the new code path.
- Update `tests/test_audio_backends_config.py` for the new
  `derive_audio_backends` signature.
- Update `tests/test_config_pipeline_mode.py` to drop `BACKEND_PROVIDER`
  env vars from its fixture (use `AUDIO_INPUT_BACKEND` /
  `AUDIO_OUTPUT_BACKEND` env vars directly).
- Update `tests/test_profile_paths.py` to monkeypatch the new dials.

### Lint / type / test pass after each commit

After Commits 1, 2, and 3: lint + tests should still pass. After Commit 4:
admin tests are reshaped. After Commit 5: frontend test (if any) reshaped.
After Commit 6: full pytest passes, full acceptance grep returns empty.

## Edge cases

- **`LocalSTTHuggingFaceRealtimeHandler` removal-of-self via base
  `_get_session_voice` fallback.** The hybrid LocalSTT* classes inherit
  `get_current_voice` from the parent (`OpenaiRealtimeHandler` /
  `HuggingFaceRealtimeHandler`), which internally calls
  `get_default_voice_for_backend(self.BACKEND_PROVIDER)`. Renamed call site
  passes `self.PROVIDER_ID` — the override value (OPENAI_BACKEND /
  HF_BACKEND) still flows through correctly.
- **`refresh_runtime_config_from_env`** has multiple sets to
  `config.BACKEND_PROVIDER` / `config.LOCAL_STT_RESPONSE_BACKEND`. All
  removed in Commit 6.
- **`_active_backend_name = get_backend_choice()`** in `LocalStream.__init__`
  is computed before the instance .env is loaded, so it's only meaningful
  on the second restart. Renaming it to `_active_provider_id` keeps the
  semantic.
- **The legacy `LocalSTTLlamaGeminiTTS*` / `LocalSTTLlamaElevenLabs*`
  classes** were already deleted in 4e. None of their references remain.
