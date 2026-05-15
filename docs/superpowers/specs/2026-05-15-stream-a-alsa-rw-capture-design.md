# Stream A — Direct ALSA RW Capture (bypass MMAP attenuation)

**Date:** 2026-05-15
**Scope:** new `src/robot_comic/audio_input/` package + `LocalStream.record_loop` wiring + one new env var in `config.py`
**Issue:** #314 (cold-boot STT stall — root cause confirmed as ALSA MMAP-mode signal attenuation on `reachymini_audio_src`)

## Background

The Reachy Mini USB audio chip's `dsnoop` endpoint, `reachymini_audio_src`, delivers ~1/10 the signal level when opened in MMAP-interleaved mode vs RW-interleaved mode. Side-by-side `arecord` runs prove it (RW peak ≈ 1.0, MMAP peak ≈ 0.1 — same device, same audio, same moment). GStreamer's `alsasrc` has no `force-rw` knob and defaults to MMAP, so every byte that flows through the daemon's audio pipeline reaches `r.media.get_audio_sample()` attenuated to room-noise floor. Moonshine correctly reports "no speech" on this signal — the stall is real, just not Moonshine's fault.

Full evidence and measurement table in [`docs/references/alsa-mmap-attenuation.md`](../../references/alsa-mmap-attenuation.md).

Per the no-upstream rule, we do not patch the daemon, GStreamer, or the XMOS driver. We open a second reader on the same dsnoop endpoint in RW mode from our app. `dsnoop` is designed for multiple readers; the daemon's MMAP reader stays running for AEC and for TTS output. The daemon's TTS playback path (`r.media.push_audio_sample`) is untouched.

## Design

### 1. Capture module — `audio_input/alsa_rw_capture.py`

A thin wrapper that spawns `arecord` as a subprocess and yields frames in the same shape `record_loop` already consumes from `r.media.get_audio_sample()`:

```
arecord -q -D reachymini_audio_src -M no -f S16_LE -r 16000 -c 2 -t raw
```

`-M no` is the explicit "no MMAP" flag (arecord defaults to RW but we set it explicitly so any future arecord default change can't silently regress us).

Public surface:

```python
class AlsaRwCapture:
    def __init__(self, device: str = "reachymini_audio_src",
                 sample_rate: int = 16000, channels: int = 2,
                 frame_samples: int = 256) -> None: ...
    def start(self) -> None: ...                  # spawns arecord
    def stop(self) -> None: ...                   # SIGTERM + wait
    def get_audio_sample(self) -> np.ndarray | None:
        """Return next (frame_samples, channels) int16 frame, or None if no data ready yet."""
    @property
    def sample_rate(self) -> int: ...
```

`get_audio_sample()` reads `frame_samples * channels * 2` bytes from the subprocess stdout in a non-blocking way (`select`/`O_NONBLOCK` on the pipe fd) and returns `np.frombuffer(buf, np.int16).reshape(frame_samples, channels)`. If stdout has fewer than one full frame buffered, return `None` — the existing `await asyncio.sleep(0)` cadence in `record_loop` keeps it non-busy.

Linux-only. On macOS/Windows, importing the module is fine, but `start()` raises `RuntimeError("AlsaRwCapture is Linux-only")`. This keeps the test suite portable.

### 2. Source-abstraction in `console.py`

Add a tiny indirection in `LocalStream`:

```python
self._audio_source = self._build_audio_source()  # in __init__ or in start()

def _build_audio_source(self):
    if config.AUDIO_CAPTURE_PATH == "alsa_rw":
        return AlsaRwCapture()  # raises on non-Linux
    return _DaemonAudioSource(self._robot)  # delegates to r.media

# in record_loop():
audio_frame = self._audio_source.get_audio_sample()
if audio_frame is not None:
    await self.handler.receive((self._audio_source.sample_rate, audio_frame))
```

`_DaemonAudioSource` is a 5-line shim that wraps `r.media` so the call sites are uniform. It lives in the same `console.py` file, no new module. Sample-rate caching mirrors the existing `input_sample_rate = self._robot.media.get_input_audio_samplerate()` call (read once at construction).

### 3. Config flag — `config.py`

One new variable, read once at module load via the existing `refresh()` pattern:

```python
AUDIO_CAPTURE_PATH: str  # "daemon" | "alsa_rw"
```

Read from `REACHY_MINI_AUDIO_CAPTURE_PATH` env var. Default:

- `"alsa_rw"` on Linux
- `"daemon"` on macOS / Windows / anything not Linux

Detection via `sys.platform == "linux"`. Operator can override either direction.

Sim mode (`--sim`) bypasses this entirely — sim constructs `LocalStream(handler=None, robot=None, ...)` for the admin UI only; no `record_loop` runs.

### 4. Lifecycle

- `AlsaRwCapture.start()` called from `LocalStream._launch_streams_with_settings` (where `start_recording()` runs today), guarded by the flag.
- `AlsaRwCapture.stop()` called from `LocalStream.stop`, alongside the existing stop paths.
- Subprocess SIGTERM on stop with a 1-second join; SIGKILL if it doesn't exit. Stderr captured to a log line on shutdown for diagnostic purposes.

### 5. Pre-wire sanity script

Before touching `record_loop`, ship a small standalone script `scripts/dsnoop_multireader_check.py` that:

1. Confirms the daemon is running (`systemctl is-active reachy-mini-daemon.service`).
2. Starts `AlsaRwCapture`, pulls ~3 seconds of frames.
3. Logs RMS / peak of those frames.
4. Logs whether `arecord` produced any stderr.

Run this on the Pi as Step 1 of the field test. If frames flow and peak ≈ 1.0 while the daemon is alive, we have multi-reader confirmation. If `arecord` errors with `device busy` or the daemon's audio pipeline reports xruns, we stop and reassess before any wiring.

This script ships as a permanent diagnostic — keep it in the tree for future regression testing per the "Retest cadence" note in [`docs/references/alsa-mmap-attenuation.md`](../../references/alsa-mmap-attenuation.md).

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/audio_input/__init__.py` | New package init (empty + `__all__`) |
| `src/robot_comic/audio_input/alsa_rw_capture.py` | New `AlsaRwCapture` class |
| `src/robot_comic/console.py` | Add `_audio_source` indirection + `_DaemonAudioSource` shim + lifecycle hooks |
| `src/robot_comic/config.py` | Add `AUDIO_CAPTURE_PATH` with platform-aware default |
| `scripts/dsnoop_multireader_check.py` | New standalone diagnostic |
| `tests/test_alsa_rw_capture.py` | Unit tests: non-Linux raises, parses raw bytes correctly, stops cleanly |

No changes to handlers, profiles, tools, or the daemon-side path. No new third-party Python deps.

## Testing

### Unit (cross-platform — runs in CI)

- `AlsaRwCapture.start()` raises `RuntimeError` on `sys.platform != "linux"`.
- Synthetic raw-bytes test: feed known S16 LE payload through a fake subprocess `Popen.stdout`, assert returned `np.ndarray` has correct shape, dtype, and values.
- `stop()` sends SIGTERM and joins; no zombie process.
- `_build_audio_source` returns `_DaemonAudioSource` when flag is `daemon` and `AlsaRwCapture` when `alsa_rw`.

### Field (on Pi — manual, gated)

1. Restore `local_stt_realtime.py` on Pi (remove `[FRAME_ENERGY]` diagnostic).
2. `git pull` + `uv pip install -e .` in `/venvs/apps_venv`.
3. `arecord -L | grep reachymini_audio_src` — confirm alias resolves.
4. `python scripts/dsnoop_multireader_check.py` — confirm RW peak ≈ 1.0 while daemon is alive.
5. Stop autostart if running. Run `python -m robot_comic.main` foreground so logs are visible.
6. Play `hello.m4a` from laptop per [`docs/references/audio-playback-recipe.md`](../../references/audio-playback-recipe.md).
7. Verify: Moonshine `on_line_completed` fires → llama-server response → ElevenLabs TTS through speaker.
8. Repeat with `my name is tony.m4a`.
9. Enable `reachy-app-autostart`, reboot the Pi, verify cold-boot path completes a turn end-to-end. Repeat reboot 3× to catch the intermittent bug pattern noted in the memory.

PR does not merge until step 9 is green at least 3 reboots in a row.

## Risks & open questions

- **AEC alignment**: dsnoop reads from `hw:0,0`, which is post-XMOS-AEC, so RW-mode should hand us the same AEC-cleaned audio as the daemon — just at proper level. To verify: in the smoke test, listen for self-hearing (robot speaks → robot transcribes its own voice). If that fires, we'll need to look at whether dsnoop slicing changes alignment relative to the daemon's reference signal. Treat as a step-9 acceptance gate.
- **Subprocess lifetime under daemon restart**: if the daemon restarts mid-session, does our `arecord` process survive (dsnoop continues to exist as long as the kernel ALSA layer does)? Manually test by SIGHUP'ing the daemon during a session and confirming our capture keeps flowing.
- **CPU/IO overhead**: 16 kHz × 2-ch × 2 bytes = 64 KB/s. Negligible. No measurement gate.
- **Replacing arecord with pyalsaaudio later**: out of scope for Stream A. `AlsaRwCapture`'s public API is small enough that we can drop in a pyalsaaudio backend without touching `console.py`.

## Success criteria

- Cold-boot path produces a full turn (mic → STT → LLM → TTS) on the first try after reboot, repeated 3× in a row.
- The pre-wire diagnostic shows RW peak ≈ 1.0 while the daemon is also reading.
- No new failures in the existing pytest suite on Linux or non-Linux dev hosts.
- `AUDIO_CAPTURE_PATH=daemon` (manual override on Pi) still works — i.e., we can fall back to the old path without code edits if something goes wrong.
