# Phase 5f — TDD plan

Companion to `docs/superpowers/specs/2026-05-16-phase-5f-faster-whisper-stt-adapter.md`.

---

## Steps (TDD order)

### 1 — Config knobs first
- Add `AUDIO_INPUT_FASTER_WHISPER` constant + `AUDIO_INPUT_CHOICES` entry
  in `src/robot_comic/config.py`.
- Extend `_SUPPORTED_AUDIO_COMBINATIONS` with the three new tuples
  (chatterbox / elevenlabs / gemini_tts).
- TEST: write unit test in `tests/test_config.py` (if it exists; else a
  small inline test in `tests/test_handler_factory.py`) asserting:
  - `AUDIO_INPUT_FASTER_WHISPER == "faster_whisper"`.
  - `"faster_whisper"` is in `AUDIO_INPUT_CHOICES`.
  - `_normalize_audio_input_backend("faster_whisper")` returns
    `"faster_whisper"` (not None and no warn).
  - `(faster_whisper, chatterbox)` is supported.

### 2 — Adapter scaffolding + Protocol conformance
- Write `tests/adapters/test_faster_whisper_stt_adapter.py` with these
  failing tests first:
  - `test_adapter_satisfies_stt_backend_protocol`.
  - `test_constructor_accepts_no_arguments`.
  - `test_constructor_accepts_should_drop_frame`.
  - `test_reset_per_session_state_is_noop`.
- Implement the adapter skeleton: bare `class FasterWhisperSTTAdapter`
  with `start` / `feed_audio` / `stop` / `reset_per_session_state`
  methods (no internals yet, just signature satisfaction).

### 3 — start() loads model + VAD on a thread
- TEST: `test_start_loads_model_and_vad_iterator` — stub the lazy
  imports, assert `WhisperModel(...)` called with `"tiny.en"` +
  `compute_type="int8"`, assert `VADIterator(...)` constructed.
- TEST: `test_start_records_callbacks` — store the callbacks for later
  dispatch.
- Implement `start()` with `asyncio.to_thread` model load + callback
  recording.

### 4 — feed_audio + VAD start event
- TEST: `test_feed_audio_starts_utterance_on_vad_start_event` — stub
  VADIterator to return `{"start": 0}` on first chunk; verify
  `on_speech_started` fires.
- TEST:
  `test_feed_audio_does_not_call_transcribe_until_vad_end_event` —
  stream chunks that only ever produce `start`; transcribe must not
  fire.
- Implement chunking + VAD-iter dispatch + speech_started callback wiring.

### 5 — feed_audio end event → transcribe → on_completed
- TEST:
  `test_feed_audio_transcribes_and_fires_on_completed_on_vad_end` —
  stub VADIterator to return `{"end": ...}` on the second chunk; stub
  WhisperModel.transcribe to return a single-segment iterator with text
  `"hello robot"`; verify `on_completed("hello robot")` fires.
- TEST:
  `test_feed_audio_joins_multi_segment_transcribe_result` — stub
  transcribe to return two segments, verify the joined text passes to
  the callback.
- TEST:
  `test_feed_audio_empty_transcribe_result_does_not_fire_callback` —
  empty / whitespace-only result is dropped (matches moonshine's
  behaviour on no-text completed events; see partial-empty-drop test).
- Implement utterance-buffer accumulation, `asyncio.to_thread` for
  transcribe, callback dispatch + drop-empty filter.

### 6 — should_drop_frame echo-guard
- TEST: `test_should_drop_frame_when_true_skips_feed_audio` — guard
  truthy → no VAD/transcribe activity.
- TEST: `test_should_drop_frame_when_false_processes_normally`.
- TEST: `test_should_drop_frame_default_none_processes_every_frame`.
- Implement the guard short-circuit at the top of `feed_audio`.

### 7 — stop() teardown
- TEST: `test_stop_clears_state_and_releases_model`.
- TEST: `test_stop_is_safe_when_never_started`.
- TEST: `test_stop_swallows_model_close_errors` (best-effort cleanup).
- Implement `stop()`.

### 8 — Callback exception handling
- TEST: `test_on_completed_exception_does_not_crash_adapter` (mirror
  of moonshine standalone test).
- TEST: `test_on_speech_started_exception_does_not_crash_adapter`.
- Implement try/except wrappers.

### 9 — Factory dispatch helper
- TEST in `tests/test_handler_factory.py`:
  - `test_faster_whisper_chatterbox_llama_routes_to_composable` —
    asserts the result is a `ComposableConversationHandler`.
  - `test_faster_whisper_chatterbox_uses_faster_whisper_adapter` —
    asserts `pipeline.stt` is a `FasterWhisperSTTAdapter`.
  - `test_faster_whisper_chatterbox_wires_should_drop_frame_callback`.
  - `test_faster_whisper_elevenlabs_uses_faster_whisper_adapter`.
  - `test_faster_whisper_gemini_tts_uses_faster_whisper_adapter`.
  - `test_moonshine_chatterbox_still_uses_moonshine_adapter`
    (regression — confirm we did not break the existing path).
- Implement `_build_stt_adapter(input_backend, should_drop_frame)` and
  swap into the five `_build_composable_*` helpers. Extend the outer
  branch in `HandlerFactory.build` to accept `AUDIO_INPUT_FASTER_WHISPER`.

### 10 — Lint / format / type / test full sweep
- `uvx ruff@0.12.0 check .` (whole repo).
- `uvx ruff@0.12.0 format --check .`.
- mypy on the new + modified files.
- `pytest tests/ -q` (modulo known main flakes).

### 11 — Dep updates
- Add `silero-vad` to a new `faster_whisper_stt` extra (keep
  `faster-whisper` in `asset_build` AND list it in the new extra so the
  STT runtime install is self-describing).
- `uv lock` to regenerate. Commit `uv.lock`.

### 12 — DEVELOPMENT.md note
- Brief A/B operating-mode block (env var + install command).

### 13 — Commit + push
- Conventional commits scoped `phase-5f`.
- Push to `claude/phase-5f-faster-whisper-stt-adapter`.
- Do NOT open the PR; return contract instead.
