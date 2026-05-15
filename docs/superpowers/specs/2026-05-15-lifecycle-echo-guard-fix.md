# Lifecycle Hook #1 — `_speaking_until` Echo-Guard Fix (Option A)

**Branch:** `claude/lifecycle-echo-guard-fix`
**Epic:** #337 (pipeline refactor) — Deferred lifecycle hooks
**Date:** 2026-05-15
**Author:** sub-agent (manager-driven)

## Background — why the original "for free" claim is false

`PIPELINE_REFACTOR.md` line 316 originally asserted that the `_speaking_until`
echo-guard timestamp would propagate "for free via adapter delegation" on the
composable path. A prior investigation proved this false.

The single runtime write site for `_speaking_until` today is
`ElevenLabsTTSResponseHandler.emit()` (line ~470–473) — and a structurally
identical write in `BaseLlamaResponseHandler.emit()` (line ~215–218). Both
derive the deadline as:

```python
cooldown_s = config.ECHO_COOLDOWN_MS / 1000.0
bytes_per_second = OUTPUT_SAMPLE_RATE * _BYTES_PER_SAMPLE
self._speaking_until = (
    self._response_start_ts + self._response_audio_bytes / bytes_per_second + cooldown_s
)
```

On the composable path, `ComposableConversationHandler.emit()` drains the
`ComposablePipeline.output_queue` (filled by the TTS adapter), not the legacy
handler's `emit()`. So `_speaking_until` stays at `0.0` for the wrapped
handler's lifetime once `FACTORY_PATH=composable`.

`LocalSTTInputMixin._handle_local_stt_event` (`local_stt_realtime.py:600-603`
and `:772-774`) reads `_speaking_until` off the handler object to suppress
transcripts caused by speaker echo. The legacy handler IS the
`_tts_handler` field on the wrapper for composable mode — so the echo guard
read keeps working as long as someone updates `_speaking_until` on that
object. The bug is that no one does.

**Impact at 4d default flip:** the robot hears itself speak and immediately
starts a new turn off its own audio output.

## Operator decision — Option A

Move the `_speaking_until` derivation to the frame-enqueue helper so it
fires on every audio frame pushed by either path (legacy `emit()` flow or
composable adapter flow). One write site per TTS path. Same formula.

## Scope

| File | Change |
|------|--------|
| `src/robot_comic/elevenlabs_tts.py` | Move `_speaking_until` derivation from `emit()` into `_enqueue_audio_frame`. Remove the now-redundant write in `emit()`. |
| `src/robot_comic/llama_base.py` | Add `_enqueue_audio_frame(frame, target_queue=None)` to `BaseLlamaResponseHandler`. It updates `_response_audio_bytes`, `_response_start_ts`, and `_speaking_until`. Route the existing `_drain_after_prev` byte-count site through it. Remove the `_speaking_until` derivation from `emit()`. |
| `src/robot_comic/llama_elevenlabs_tts.py` | Route the three direct `out_queue.put((rate, frame))` sites in `_synthesize_and_enqueue` / `_stream_tts_to_queue` through `_enqueue_audio_frame(frame, target_queue=out_queue)`. Otherwise the composable `(moonshine, llama, elevenlabs)` triple stays broken. |
| `src/robot_comic/chatterbox_tts.py` | Route the two direct `output_queue.put((rate, frame))` sites in `_synthesize_and_enqueue` through `_enqueue_audio_frame(frame)`. Otherwise the composable `(moonshine, llama, chatterbox)` and `(moonshine, gemini, chatterbox)` triples stay broken. |
| `tests/test_echo_suppression.py` | Update the existing two `test_emit_sets_speaking_until_from_byte_count` tests to assert against `_enqueue_audio_frame` instead of `emit()` (the new single source of truth). |
| `tests/test_composable_echo_guard.py` | NEW — two regression tests exercising `_speaking_until` updates through `ElevenLabsTTSAdapter` for both the `ElevenLabsTTSResponseHandler` and `LlamaElevenLabsTTSResponseHandler` wrapped paths. |

## Files NOT touched

- `src/robot_comic/llama_gemini_tts.py` — host of `LocalSTTLlamaGeminiTTSHandler`, confirmed unreachable from the factory today (memo: phase 4e orphan list). Touching it would expand scope without changing observable behaviour.
- `src/robot_comic/gemini_tts.py` — bundled Gemini TTS handler has no echo guard today (`_speaking_until` is not defined on the class). Pre-existing limitation outside this hook's scope.
- `src/robot_comic/composable_conversation_handler.py` — unchanged.
- `src/robot_comic/composable_pipeline.py` — unchanged.
- Adapter Protocols — no change to the four-member duck-typed surface.
- `ConversationHandler` ABC — unchanged.
- `handler_factory.py` — unchanged.

## Formula equivalence

Both fix sites must produce the same `_speaking_until` value the legacy
`emit()` site produces today. The two sites keep their respective
module-level constants (`ELEVENLABS_OUTPUT_SAMPLE_RATE` vs
`_OUTPUT_SAMPLE_RATE`, both 24 kHz). The arithmetic is identical.

## Acceptance criteria

- Two new regression tests in `tests/test_composable_echo_guard.py` pass with the fix; both fail without it.
- All existing tests in `tests/test_echo_suppression.py` still pass (updated to track the new write site).
- All existing tests in `tests/test_llama_elevenlabs_tts.py`, `tests/test_elevenlabs_tts.py`, `tests/test_chatterbox_tts.py`, `tests/adapters/` still pass.
- `uvx ruff@0.12.0 check` and `format --check` green from repo root.
- `.venv/bin/mypy --pretty <changed files>` green.
- `.venv/bin/pytest tests/ -q` green.
- `_speaking_until` derivation lives in exactly one site per TTS path. Legacy `emit()` write sites removed.

## Out of scope

- Wiring echo guard into `gemini_tts.py` or `llama_gemini_tts.py`.
- Plumbing tags or first-audio markers through the composable adapters.
- Changing adapter Protocols or `ConversationHandler` ABC.
