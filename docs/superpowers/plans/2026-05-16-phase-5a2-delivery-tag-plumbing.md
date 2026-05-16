# Phase 5a.2 — TDD plan

**Spec:** `docs/superpowers/specs/2026-05-16-phase-5a2-delivery-tag-plumbing.md`
**Branch:** `claude/phase-5a-2-delivery-tag-plumbing`
**Commit prefix:** `phase-5a-2`

Mirror the Phase 4 cadence: one failing test → minimum implementation → green → commit.

## Task 1 — `LLMResponse.delivery_tags` field

- Add failing test in `tests/test_backends_protocols.py`:
  - `test_llm_response_has_delivery_tags_default_empty_tuple` — `LLMResponse().delivery_tags == ()`.
  - `test_llm_response_accepts_delivery_tags` — `LLMResponse(delivery_tags=("fast",)).delivery_tags == ("fast",)`.
- Implement in `src/robot_comic/backends.py`: add `delivery_tags: tuple[str, ...] = ()` to `LLMResponse` with docstring update.
- Verify the three LLM adapters' existing `LLMResponse(...)` calls compile (they should — default value).
- Commit: `feat(phase-5a-2): add LLMResponse.delivery_tags channel`

## Task 2 — `TTSBackend.synthesize` first-audio-marker channel (Protocol + mock)

- Add failing test in `tests/test_backends_protocols.py`:
  - `test_tts_protocol_synthesize_accepts_first_audio_marker_kwarg` — call the `_MockTTS.synthesize(text, first_audio_marker=marker)` and assert no TypeError. Marker stays empty because the mock doesn't append.
- Update `_MockTTS.synthesize` signature in the test to accept `first_audio_marker: list[float] | None = None`.
- Implement in `src/robot_comic/backends.py`: extend `TTSBackend.synthesize` Protocol signature with `first_audio_marker: list[float] | None = None`. Docstring update.
- Verify all three TTS adapter `synthesize` methods still satisfy the Protocol (Protocol is `runtime_checkable` but only checks method *names*; signature compat is mypy-only). After this task adapters will fail mypy until tasks 3-5 add the param — acceptable mid-TDD.
- Commit: `feat(phase-5a-2): add first_audio_marker channel to TTSBackend Protocol`

## Task 3 — `ElevenLabsTTSAdapter.synthesize` records first-audio marker

- Failing tests in `tests/adapters/test_elevenlabs_tts_adapter.py`:
  - `test_synthesize_appends_first_audio_marker_on_first_frame` — pass `marker = []`, drive the stub to yield 3 frames, assert `len(marker) == 1` after iteration completes.
  - `test_synthesize_marker_is_only_appended_once` — same setup, assert append happens exactly once.
  - `test_synthesize_does_not_touch_marker_when_none` — confirm the default-None path is a no-op.
- Implement: add `first_audio_marker: list[float] | None = None` param; append `time.monotonic()` to it on the first frame yielded.
- Commit: `feat(phase-5a-2): ElevenLabsTTSAdapter populates first_audio_marker`

## Task 4 — `GeminiTTSAdapter.synthesize` consume-or-fallback + first-audio marker

- Failing tests in `tests/adapters/test_gemini_tts_adapter.py`:
  - `test_synthesize_consumes_delivery_tags_param_when_non_empty` — pass `tags=("fast",)` with plain text (no markers); assert the `system_instruction` arg of the TTS call contains "Delivery cues for this line:" suffix (proving `build_tts_system_instruction` received non-empty tags from the param, not from text).
  - `test_synthesize_falls_back_to_text_parsing_when_tags_empty` — existing behaviour; pass `tags=()` with `[fast] Plain text.`; assert delivery cues suffix appears.
  - `test_synthesize_appends_first_audio_marker_on_first_frame` + `test_synthesize_does_not_touch_marker_when_none`.
  - Update existing `test_synthesize_ignores_protocol_tags_arg` — the old test asserts that `tags=("fast",)` is ignored; this is now wrong. Replace with the new consume-or-fallback test above.
- Implement: replace `del tags`. Compute `effective_tags` once: `if tags: effective_tags = list(tags); else: per-sentence extract_delivery_tags fallback`. The `SHORT_PAUSE_TAG` handling stays per-sentence in the fallback path; in the consume-from-param path it triggers a single pre-stream silence frame iff `SHORT_PAUSE_TAG in tags`. Add `first_audio_marker` param + appender at first yield.
- Commit: `feat(phase-5a-2): GeminiTTSAdapter consumes delivery tags + populates marker`

## Task 5 — `ChatterboxTTSAdapter.synthesize` first-audio marker + tags DEBUG log

- Failing tests in `tests/adapters/test_chatterbox_tts_adapter.py`:
  - `test_synthesize_appends_first_audio_marker_on_first_frame` + `test_synthesize_does_not_touch_marker_when_none`.
  - `test_synthesize_logs_non_empty_tags_at_debug` — pass `tags=("fast",)`, assert a DEBUG-level log mentioning tag forwarding.
- Implement: add `first_audio_marker: list[float] | None = None`; on non-empty `tags`, `logger.debug("ChatterboxTTSAdapter: dropping delivery tags %r; legacy handler reads from active persona", tags)`. Append marker on first yield. Update module docstring "Known gaps" bullet to point at this spec.
- Commit: `feat(phase-5a-2): ChatterboxTTSAdapter logs dropped tags + populates marker`

## Task 6 — `ComposablePipeline._speak_assistant_text` threads both channels

- Failing tests in `tests/test_composable_pipeline.py`:
  - Update `_RecordingTTS.synthesize` signature in the test to accept `first_audio_marker: list[float] | None = None`; record the marker reference.
  - `test_speak_assistant_text_threads_delivery_tags_to_tts` — script an LLM response with `LLMResponse(text="Hi!", delivery_tags=("fast",))`; assert the TTS mock's recorded call has `tags == ("fast",)`.
  - `test_speak_assistant_text_allocates_first_audio_marker` — assert the TTS mock received a non-None list. (Population is a per-adapter responsibility, asserted in tasks 3-5.)
- Implement: in `_speak_assistant_text`, allocate `marker: list[float] = []`; call `self.tts.synthesize(text, tags=response.delivery_tags, first_audio_marker=marker)`. Remove the long TODO block; replace with a one-line pointer to this spec.
- Commit: `feat(phase-5a-2): orchestrator threads delivery_tags + first_audio_marker into TTS`

## Task 7 — Lint / format / type / test from repo root

- Run from repo root:
  ```
  uvx ruff@0.12.0 check .
  uvx ruff@0.12.0 format --check .
  .venv/Scripts/mypy --pretty --show-error-codes src/robot_comic/backends.py src/robot_comic/composable_pipeline.py src/robot_comic/adapters/
  .venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
  ```
- Fix any drift surfaced. If the test suite reveals a downstream consumer (e.g. `test_composable_persona_reset.py`) calling `tts.synthesize(text)` positionally and now broken — they shouldn't, all adapter call-sites are kwarg-style, but verify.
- Commit (if anything emerges): `chore(phase-5a-2): lint/format adjustments`

## Task 8 — Push branch + manager opens PR

- `git push -u origin claude/phase-5a-2-delivery-tag-plumbing`.
- Return summary to manager.
