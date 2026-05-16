# Phase 5f.1 — swap silero-vad for webrtcvad in `FasterWhisperSTTAdapter`

**Date:** 2026-05-16
**Status:** Spec — implementation on `claude/phase-5f-1-webrtcvad-swap`.
**Tracks:** epic #391; follow-up to PR #416 (Phase 5f).
**Predecessor:** Phase 5f (`#416`) — `FasterWhisperSTTAdapter` shipped with
`silero-vad` as the chunker. Install size blows out the on-robot eMMC.

---

## §1 — Problem

PR #416 wired faster-whisper as the alternate STT and reached for
`silero-vad` (the standard "neural VAD" pick) to chunk streaming audio into
utterances because faster-whisper is batch-only and needs explicit
utterance boundaries.

Tried on ricci (the chassis) today (2026-05-16):

```
$ uv pip install -e .[faster_whisper_stt]
...
× Failed to extract archive: torch-2.5.1-cp312-cp312-linux_aarch64.whl
  └─ No space left on device
```

`silero-vad` transitively depends on `torch` (~2 GB unpacked). ricci has a
14 GB eMMC with ~1.4 GB free post-base-install. The whole
`[faster_whisper_stt]` extra was supposed to ship the tiny.en model
(~75 MB) plus CT2 runtime plus a VAD — total budget ~500 MB. silero-vad's
torch dep on its own busts the budget by 4×.

5f was specced before we knew the eMMC was this tight. The hardware
constraint is fixed; the VAD choice has to give.

## §2 — Decision

**Replace silero-vad with `webrtcvad`** (Google's WebRTC VAD wrapped as a
Python C extension).

| Option | Why considered | Why rejected / picked |
|---|---|---|
| **`webrtcvad`** | Pure C extension, ~50 KB wheel, no torch / no onnx. Mature: 10+ years on PyPI. Used in countless ASR pipelines. Simple `is_speech(frame_bytes, sample_rate) -> bool` API per 10/20/30 ms frame at 8/16/32/48 kHz. **Picked.** |
| Energy-based VAD (RMS threshold) | Zero deps, ~20 LoC. Less robust in noisy environments. Defer unless webrtcvad turns out to misbehave on chassis mics. |
| Keep silero-vad behind `[faster_whisper_stt_pc]` extra for laptops | Splits the matrix; only solves the install gap on chassis. Operator preference is one canonical extra. |

webrtcvad is operator's pick (see hardware findings in
`docs/superpowers/memory/`).

## §3 — Adapter design with webrtcvad

webrtcvad's contract:

- Constructor: `vad = webrtcvad.Vad(aggressiveness)` where
  aggressiveness is 0–3 (higher = more aggressive at filtering
  non-speech). We default to **mode 2** — a middle ground; on-device
  tuning can follow.
- Per-frame check: `vad.is_speech(frame: bytes, sample_rate: int) -> bool`.
  `frame` must be **exactly 10/20/30 ms of int16 PCM at 8/16/32/48 kHz**.
- No state to load — instantaneous construction, no model file.

### Frame sizing

Pick **16 kHz, 30 ms frames = 480 samples = 960 bytes** (int16). 30 ms is
the longest webrtcvad accepts, which minimises per-frame call overhead and
gives the silence-counter the most fan-out for a given latency budget.

### Utterance boundary heuristic

silero-vad's `VADIterator` emits explicit `{"start"}` / `{"end"}` events.
webrtcvad just gives `True`/`False` per frame. We synthesise the same
events with a tiny state machine:

- Track consecutive `True` and `False` frame counts.
- **Start** an utterance once **3 consecutive speech frames** are observed
  (≈90 ms) — debounces single-frame noise bursts. Fires
  `on_speech_started()`.
- **End** the utterance once **17 consecutive non-speech frames** are
  observed (≈510 ms of silence). Submits the accumulated buffer to
  faster-whisper.

These thresholds match what production speech pipelines typically use for
webrtcvad mode 2 (e.g. py-webrtcvad's own README example uses 30 ms
frames + 300 ms trailing silence; we widen to 510 ms because the comedian
persona deliberately pauses for effect).

A class-level pair of constants makes them easy to tune from a single
place without touching the state machine.

### Audio frame conversion

Same int16-PCM → 16 kHz mono path the silero version had, **but**:

1. webrtcvad consumes `bytes` not float32, so we skip the float
   normalisation step.
2. The chunk size goes from 512 (silero) to 480 (webrtcvad). The
   surrounding chunker / `_chunk_remainder` accounting is otherwise
   identical.

### Public surface stays unchanged

- `start(on_completed, on_partial=None, on_speech_started=None)` — same.
- `feed_audio(frame)` — same.
- `stop()` — same; no model to `.close()`, but we still clear state.
- `reset_per_session_state()` — same no-op.
- `should_drop_frame` echo-guard — same.
- `on_partial` still documented as never-fired (faster-whisper is batch).
- `FasterWhisperSTTDependencyError` still raised on missing deps; message
  now points at the same install command.
- `STTBackend` Protocol conformance preserved.

## §4 — `pyproject.toml`

```toml
faster_whisper_stt = [
    # Phase 5f / 5f.1: faster-whisper STT as alternate to Moonshine.
    "faster-whisper>=1.0.0",
    # webrtcvad chunks streaming audio into utterances; faster-whisper is
    # batch-only. Picked over silero-vad because silero-vad pulls torch
    # (~2 GB) which won't fit on the 14 GB eMMC.
    "webrtcvad>=2.0.10",
]
```

`uv lock` regenerated. CI's `uv-lock-check` is gating; been bitten by
forgetting this three times this session — re-emphasising.

## §5 — Test strategy

The existing 32-test suite at
`tests/adapters/test_faster_whisper_stt_adapter.py` stubs silero-vad and
faster-whisper via `sys.modules` injection. Swap the silero stub for a
webrtcvad stub:

- Stub `webrtcvad.Vad(aggressiveness)` returns an object with
  `set_mode(mode)` and `is_speech(frame_bytes, sample_rate) -> bool`.
- Tests script `is_speech` returns as a list of bools (one per
  consecutive call), mirroring how the silero tests scripted
  `VADIterator` events.

The state-machine threshold (3 speech to start, 17 silence to end) is
honoured by stubbing **enough consecutive True / False frames to cross
the boundary**. Tests parameterise this via small helpers.

Behavioural assertions preserved across the swap:

- start → model + VAD load.
- start idempotent / re-bindable.
- start raises `FasterWhisperSTTDependencyError` when webrtcvad missing.
- feed_audio chunks 1024-sample frame into two 480-sample VAD calls
  (the third 64-sample remainder stays in `_chunk_remainder`).
- A run of speech frames → `on_speech_started()` fires once.
- A run of silence frames after speech → transcribe + `on_completed`.
- Back-to-back utterances → two completions.
- Empty / whitespace transcript dropped.
- `should_drop_frame=True` short-circuits before the VAD call.
- `stop()` is idempotent and safe-when-never-started.
- Callbacks raising do not crash the adapter.
- Transcribe raising does not crash the adapter.
- Protocol conformance via `isinstance(adapter, STTBackend)`.

Coverage target: parity with the 32 silero tests modulo a small
delta for tests that were silero-API-specific (e.g. `reset_states`,
which webrtcvad does not have).

## §6 — Install-size estimate

| Component | Approx size |
|---|---|
| `faster-whisper` wheel + CTranslate2 | ~80 MB |
| `ctranslate2` runtime libs | ~140 MB |
| `webrtcvad` wheel | <100 KB |
| `tiny.en` model (first run download) | ~75 MB |
| `numpy` / `scipy` (already in base) | 0 (already installed) |
| **Total incremental** | **~300 MB** |

vs the silero version which transitively needed:

| Component | Approx size |
|---|---|
| `silero-vad` wheel | ~5 MB |
| `torch` wheel (~2 GB unpacked) | **~800 MB on-disk** |
| `torchaudio` | ~50 MB |
| `onnxruntime` (transitive) | ~70 MB |
| **Total incremental** | **~1.0 GB+ (FAILED on ricci)** |

The new total fits the ~500 MB budget with headroom for model warm-up
state.

## §7 — Out of scope

- Any other adapter (Moonshine untouched).
- Adding energy-based VAD fallback.
- Tuning aggressiveness / start-debounce / silence-end thresholds in the
  field — defer to on-device A/B once installed.
- Re-architecting `feed_audio` chunk accounting beyond the 512→480 swap.
- Telemetry / on-device benchmarks.
- Touching the standalone STT shape or the `STTBackend` Protocol.

## §8 — Acceptance

- `uvx ruff@0.12.0 check .` clean (whole repo).
- `uvx ruff@0.12.0 format --check .` clean.
- `mypy --pretty --show-error-codes src/robot_comic/adapters/faster_whisper_stt_adapter.py` clean.
- Adapter test suite green; coverage parity with 5f.
- `uv lock` regenerated; lock no longer references `silero-vad`,
  `torch`, `torchaudio`. lock adds `webrtcvad`.
- DEVELOPMENT.md extras table updated.

## §9 — Risk + rollback

- **Risk**: webrtcvad mode 2 too aggressive on the chassis mic
  post-Stream-A gain. Mitigation: aggressiveness is a single constant;
  on-device A/B can dial it. Operator validates on chassis before
  flipping the default.
- **Risk**: 30 ms / 480-sample frame size interacts with our
  `_chunk_remainder` math. Mitigation: tests cover odd-sized incoming
  frames + resampling.
- **Rollback**: revert this PR. 5f's silero version returns and the
  on-laptop install path still works (just won't fit on chassis).
