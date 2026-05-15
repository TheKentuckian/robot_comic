# Implementation Plan — Lifecycle Echo-Guard Fix

**Goal:** Move `_speaking_until` derivation from `emit()` into the
frame-enqueue helper for both TTS paths (`elevenlabs_tts.py` and
`llama_base.py`) so it fires under the composable factory path.

**Spec:** `docs/superpowers/specs/2026-05-15-lifecycle-echo-guard-fix.md`

## TDD task breakdown

### Task 1 — Composable regression tests (RED first)

- Write `tests/test_composable_echo_guard.py` with two tests:
  - `test_elevenlabs_adapter_updates_speaking_until_on_streamed_audio` — stub HTTP via `_FakeStreamCM`/`_FakeStreamResponse`, wrap `ElevenLabsTTSResponseHandler` in `ElevenLabsTTSAdapter`, drain `synthesize()`, assert `_speaking_until` > mocked now.
  - `test_llama_elevenlabs_adapter_updates_speaking_until_on_streamed_audio` — same shape against `LlamaElevenLabsTTSResponseHandler`.
- Both tests must currently FAIL (`_speaking_until` stays at `0.0`).
- Commit: `test(echo-guard): add failing regression tests for composable path (#337)`

### Task 2 — `_enqueue_audio_frame` fix in `elevenlabs_tts.py`

- In `_enqueue_audio_frame`, after the byte-count + start-ts updates, derive `_speaking_until` from the cumulative-bytes formula using the same constants as the legacy `emit()` site.
- Remove the `_speaking_until` derivation from `emit()`.
- Update `tests/test_echo_suppression.py::TestElevenLabsTTSEchoGuard::test_emit_sets_speaking_until_from_byte_count` to drive `_enqueue_audio_frame`.
- The elevenlabs regression test in `test_composable_echo_guard.py` must now GREEN.
- Commit: `fix(echo-guard): write _speaking_until from _enqueue_audio_frame in ElevenLabs handler (#337)`

### Task 3 — `_enqueue_audio_frame` helper in `llama_base.py` + subclass put-site re-routing

- Add `async def _enqueue_audio_frame(self, frame, target_queue=None)` to `BaseLlamaResponseHandler`. Mirrors the elevenlabs helper.
- Replace inline byte-count math in `_drain_after_prev` with `_enqueue_audio_frame`.
- Remove `_speaking_until` derivation from `BaseLlamaResponseHandler.emit()`.
- Route the three `out_queue.put((rate, frame))` sites in `llama_elevenlabs_tts.py` through `await self._enqueue_audio_frame(frame, target_queue=out_queue)`.
- Route the two `output_queue.put((_OUTPUT_SAMPLE_RATE, frame))` sites in `chatterbox_tts.py::_synthesize_and_enqueue` through `await self._enqueue_audio_frame(frame)`.
- Update `tests/test_echo_suppression.py::TestLlamaBaseEchoGuard::test_emit_sets_speaking_until_from_byte_count` to drive `_enqueue_audio_frame`.
- All llama / chatterbox suites stay green; the llama regression in `test_composable_echo_guard.py` goes GREEN.
- Commit: `fix(echo-guard): write _speaking_until from _enqueue_audio_frame in llama base + subclasses (#337)`

### Task 4 — Full-suite lint / mypy / pytest + push

- `uvx ruff@0.12.0 check` from repo root.
- `uvx ruff@0.12.0 format --check` from repo root.
- `.venv/bin/mypy --pretty src/robot_comic/elevenlabs_tts.py src/robot_comic/llama_base.py src/robot_comic/llama_elevenlabs_tts.py src/robot_comic/chatterbox_tts.py`.
- `.venv/bin/pytest tests/ -q`.
- `git push -u origin claude/lifecycle-echo-guard-fix`.
