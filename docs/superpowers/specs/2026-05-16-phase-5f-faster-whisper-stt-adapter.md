# Phase 5f — `FasterWhisperSTTAdapter` as alternate STT to Moonshine

**Date:** 2026-05-16
**Status:** Spec — implementation on
`claude/phase-5f-faster-whisper-stt-adapter`.
**Tracks:** epic #391; closes issue #387.
**Predecessor:** Phase 5e.6 (`#413`) — all five composable triples now build
on the standalone `MoonshineSTTAdapter()` shape. The factory's STT-adapter
construction is the only swap-point we need.

---

## §1 — Problem

Issue #387 + the 2026-05-16 STT memo
(`docs/superpowers/specs/2026-05-16-moonshine-reliability-alternates-sibling-daemon.md`)
gave **low confidence** that Moonshine's "rearm-N-then-die" stall can be
permanently fixed on our side. The memo's top alternate recommendation is
**faster-whisper `tiny.en` (CTranslate2 int8)** — ~2s cold-load (vs
Moonshine's 20s), mature C++ runtime with no equivalent stall pattern, and
slots into the existing `STTBackend` Protocol surface.

Phase 5e finished the standalone-adapter pattern: every composable triple's
factory builder constructs a `MoonshineSTTAdapter()` directly. There is no
`LocalSTTInputMixin` plumbing to share for a new STT — we just need a second
adapter class implementing `STTBackend` and a one-line swap in the five
factory builders.

## §2 — Scope (this PR)

1. **`FasterWhisperSTTAdapter`** at
   `src/robot_comic/adapters/faster_whisper_stt_adapter.py`. Implements
   the `STTBackend` Protocol post-5e:
   - `start(on_completed, on_partial=None, on_speech_started=None)`.
   - `feed_audio(frame)`.
   - `stop()`.
   - `reset_per_session_state()` (no-op for this backend).
   - Constructor takes `should_drop_frame: Callable[[], bool] | None`
     for the echo-guard skip — same shape as `MoonshineSTTAdapter`.
   - Wraps `faster_whisper.WhisperModel("tiny.en", compute_type="int8")`.
   - Uses `silero-vad` to chunk streaming frames into utterance segments.
   - Faster-whisper is batch-only: on VAD utterance-end, the accumulated
     buffer is submitted to `WhisperModel.transcribe` on a thread (via
     `asyncio.to_thread`) so the asyncio loop stays unblocked. The result
     text fires `on_completed(text)`.
   - **Partials**: silero-vad's `start` event fires `on_speech_started()`.
     We do not emit `on_partial` — faster-whisper is batch and the
     orchestrator's only consumer of `on_partial` today is the
     `user_partial` output-queue publish, which is decorative. Documented
     limitation. A future revision can sliding-window the buffer if
     partial transcripts become load-bearing.

2. **Factory wiring** in `src/robot_comic/handler_factory.py`:
   - Add helper `_build_stt_adapter(input_backend, should_drop_frame)` that
     dispatches to `MoonshineSTTAdapter` or `FasterWhisperSTTAdapter` based
     on `input_backend`.
   - All five `_build_composable_*` helpers call the new helper instead of
     instantiating `MoonshineSTTAdapter` directly. No other change to the
     builders.
   - Extend the moonshine-branch arm to accept the new
     `AUDIO_INPUT_FASTER_WHISPER` value: the entire branch body (the
     llama/gemini routing matrix) applies identically — only the STT
     adapter differs. The cleanest way is to gate the outer branch on
     "is this a local-STT input?" and let `_build_stt_adapter` switch
     internally.
   - Add `AUDIO_INPUT_FASTER_WHISPER` to the supported-combinations
     matrix in `config.py` for each existing TTS output.

3. **Config plumbing** in `src/robot_comic/config.py`:
   - Add `AUDIO_INPUT_FASTER_WHISPER = "faster_whisper"` alongside
     `AUDIO_INPUT_MOONSHINE`.
   - Add to `AUDIO_INPUT_CHOICES` so the env-var validator accepts it.
   - Add `(AUDIO_INPUT_FASTER_WHISPER, AUDIO_OUTPUT_*)` rows to
     `_SUPPORTED_AUDIO_COMBINATIONS` for chatterbox / elevenlabs /
     gemini_tts (matching the moonshine matrix).

4. **`pyproject.toml`**:
   - Promote `faster-whisper>=1.0.0` from `asset_build` into a new
     `faster_whisper_stt` extra (so the runtime STT is named clearly and
     `asset_build` keeps its meaning for build-time scripts).
     `asset_build` retains the dep so build scripts continue to work.
   - Add `silero-vad>=5.0.0` to the new extra (silero-vad reached v5 in
     2024; we want the modernised package).
   - Regenerate `uv.lock`.

5. **Tests**:
   - Per-adapter unit tests at
     `tests/adapters/test_faster_whisper_stt_adapter.py`. Stub
     `WhisperModel` and the silero-vad iterator/loader. Cover:
     - Standalone construction (no host handler argument exists at all).
     - `start` subscribes the callbacks and triggers a model-load
       (synchronous in a thread via `asyncio.to_thread`).
     - `feed_audio` → VAD chunking → model.transcribe → `on_completed`
       callback.
     - `should_drop_frame` echo-guard pattern.
     - `stop()` tears down state; idempotent / safe-when-never-started.
     - `reset_per_session_state` is a no-op.
     - Protocol conformance (`isinstance(STTBackend)`).
     - `on_speech_started` fires on VAD start event.
   - Factory smoke tests in `tests/test_handler_factory.py` for one
     `(faster_whisper, *, *)` triple (chatterbox + llama) — mirror the
     5e.* moonshine-triple patterns. Covers the new helper and dispatch.

6. **Documentation**:
   - This spec.
   - TDD plan at
     `docs/superpowers/plans/2026-05-16-phase-5f-faster-whisper-stt-adapter.md`.
   - A/B operating mode noted in `DEVELOPMENT.md`.

## §3 — Out of scope

- Removing Moonshine. Both backends live side-by-side; default stays
  Moonshine.
- Silero-VAD parameter tuning (defaults are fine; on-device tuning can
  follow in a follow-up).
- Faster-whisper parameter tuning beyond `"tiny.en"` + `compute_type="int8"`.
- Sliding-window partial transcripts (documented limitation; defer).
- `LocalSTT*RealtimeHandler` hybrids (the `(moonshine, openai_realtime_output)`
  and `(moonshine, hf_output)` paths use `LocalSTTInputMixin` directly,
  not the adapter — out of scope per Phase 4c-tris Option B). No
  `(faster_whisper, openai_realtime_output)` / `(faster_whisper, hf_output)`
  variants ship in this PR.
- Telemetry parity with Moonshine (`MOONSHINE_DIAG` etc.). Add later if
  needed.

## §4 — A/B operating mode

To run the new adapter on a chassis:

```bash
uv pip install -e .[faster_whisper_stt]
export REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper
export REACHY_MINI_AUDIO_OUTPUT_BACKEND=chatterbox   # or elevenlabs / gemini_tts
python -m robot_comic.main
```

To revert to Moonshine:

```bash
export REACHY_MINI_AUDIO_INPUT_BACKEND=moonshine
```

(or unset and rely on the default).

## §5 — Adapter internals — design notes

### Model + VAD load timing

Loading happens inside `start()` on a thread (`asyncio.to_thread`). For
faster-whisper:

```python
self._model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
```

For silero-vad — use the `silero_vad` package's `load_silero_vad()` and the
`VADIterator` helper class (silero-vad v5 ships this). The iterator works
chunk-by-chunk: pass 512 samples at 16 kHz; it returns `{"start": …}` or
`{"end": …}` dicts at utterance boundaries.

### Audio frame conversion

`STTBackend.feed_audio` receives an `AudioFrame(samples, sample_rate)`.
silero-vad expects 16 kHz float32 mono. We:
1. Convert int16 / list samples to float32 normalised in [-1, 1].
2. Resample (scipy.signal.resample) to 16 kHz if the incoming frame is at
   a different rate.
3. Feed into a chunker that splits / accumulates into 512-sample windows.
4. Pass each 512-sample chunk through `VADIterator`.
5. On `start` event → fire `on_speech_started()`, start accumulating into
   the utterance buffer.
6. On `end` event → submit accumulated buffer to `WhisperModel.transcribe`
   via `asyncio.to_thread`, fire `on_completed(text)` with the joined
   segment text.

### Thread safety

faster-whisper's `transcribe` is a sync call. We dispatch via
`asyncio.to_thread` so the asyncio loop keeps draining. The
`should_drop_frame` echo-guard short-circuits BEFORE any VAD processing,
so TTS-playback frames are silently dropped without polluting the
utterance buffer.

### What if `silero-vad` doesn't load

`silero-vad` requires `onnxruntime` (already a transitive dep of
moonshine-voice). If the optional extra isn't installed, the adapter's
`start()` raises an `ImportError` with a hint to install
`.[faster_whisper_stt]` — same shape as Moonshine's
`LocalSTTDependencyError`.

### What if faster-whisper can't emit partial transcripts

It can't, by design. We do NOT fake partials. `on_partial` is documented
as never firing for this adapter. Down-line consumers (today: only the
orchestrator's `user_partial` output-queue publish) tolerate `None`
already since the pre-5e.2 path also didn't subscribe.

## §6 — Test strategy

### Adapter unit tests

Stub the model + VAD by monkey-patching the module-level imports the
adapter performs lazily inside `start()`. The adapter never reaches into
faster-whisper or silero-vad at import time (mirror of MoonshineListener),
so the unit tests can run without either dependency installed.

Stubs simulate:
- `WhisperModel.transcribe(audio)` → returns a `(segments_iter, info)`
  pair where `segments_iter` yields objects with a `.text` attribute.
- `VADIterator(chunk)` → returns `{"start": idx}` / `{"end": idx}` /
  `None`.

### Factory smoke test

Same shape as the 5e.* tests: build the handler with
`AUDIO_INPUT_FASTER_WHISPER + AUDIO_OUTPUT_CHATTERBOX + LLM_BACKEND=llama`,
assert the resulting `ComposableConversationHandler` has a
`FasterWhisperSTTAdapter` in `pipeline.stt` and that the
`should_drop_frame` closure is wired.

### Protocol conformance

`isinstance(adapter, STTBackend)` smoke test (mirror of the moonshine one).

## §7 — Acceptance

- `uvx ruff@0.12.0 check .` clean.
- `uvx ruff@0.12.0 format --check .` clean.
- `mypy src/robot_comic/adapters/faster_whisper_stt_adapter.py
   src/robot_comic/handler_factory.py src/robot_comic/config.py` clean.
- `pytest tests/ -q` green (modulo the known main flakes).
- `uv lock` regenerated and committed.
- Factory smoke test for `(faster_whisper, llama, chatterbox)` passes.
- A/B operating-mode env-var documented in `DEVELOPMENT.md`.

## §8 — Risk + rollback

- **Risk**: faster-whisper or silero-vad isn't actually loadable on the
  Pi 5 → adapter `start()` raises, app fails to boot. Mitigation:
  default `AUDIO_INPUT_BACKEND` stays `moonshine`; operator opts in
  explicitly.
- **Risk**: silero-vad v5 API differs from what we wrote against.
  Mitigation: stub-only tests in CI; the operator validates on chassis
  in the A/B window before flipping the default.
- **Rollback**: revert this PR. Moonshine stays untouched.
