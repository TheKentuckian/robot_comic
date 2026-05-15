# ALSA MMAP attenuation on `reachymini_audio_src`

Reading from the `reachymini_audio_src` ALSA device (a `dsnoop` over `hw:0,0`,
2-ch S16_LE 16 kHz) in MMAP-interleaved mode delivers approximately **1/10 the
signal level** of the same device read in RW-interleaved mode. GStreamer's
`alsasrc` defaults to MMAP and exposes no override, so any GStreamer-based
capture pipeline silently produces near-silent audio. This is the root cause
of the `#314` cold-boot STT stall and the reason
[`audio_input/alsa_rw_capture.py`](../../src/robot_comic/audio_input/alsa_rw_capture.py)
exists.

## Measurements (2026-05-15)

All five readers running against the **same** dsnoop endpoint with the same
ambient room audio produced wildly different signal levels purely based on
the kernel-side access mode:

| Reader                                                       | Access mode           | Peak           | RMS            | Notes                                |
|--------------------------------------------------------------|-----------------------|----------------|----------------|--------------------------------------|
| `arecord -D reachymini_audio_src -f S16_LE`                  | RW-interleaved        | **32768 (1.0)**| 4121 (0.126)   | Default arecord behaviour            |
| `arecord -D reachymini_audio_src -M -f S16_LE`               | MMAP-interleaved      | 3168 (0.097)   | 588 (0.018)    | `-M` forces MMAP                     |
| `gst-launch alsasrc device=reachymini_audio_src`             | MMAP (default)        | 2491 (0.076)   | 517 (0.016)    | GStreamer's default                  |
| `ReachyMini().media.get_audio_sample()`                      | MMAP (via GStreamer)  | 0.04 (F32LE)   | 0.014          | What the comic app saw pre-Stream A  |
| `AlsaRwCapture` (`audio_input/alsa_rw_capture.py`)           | RW-interleaved        | **32768 (1.0)**| matches RW row | Our app-side workaround              |

RW is correct, MMAP is broken on this hardware. The XMOS USB-Audio
driver + kernel ALSA combo on the Reachy Mini Pi under-delivers when the
dsnoop endpoint is opened in MMAP mode. Direct `hw:0,0` MMAP wasn't
testable because the daemon owns the device exclusively at that level.

## How this killed STT

1. GStreamer's `alsasrc` element defaults to MMAP — `gst-inspect-1.0
   alsasrc` confirms there is no `access` / `use-mmap` / `force-rw`
   property.
2. The daemon's audio capture pipeline uses `alsasrc
   device=reachymini_audio_src` → MMAP → attenuated F32LE samples →
   `appsink`.
3. The comic app called `r.media.get_audio_sample()` and received
   attenuated frames (peak ≈ 0.04).
4. Moonshine received float frames that looked like the room-noise floor,
   correctly reported "no speech," and stalled indefinitely with
   `state=idle last_event=None` even after thousands of frames were
   ingested.
5. Symptom: GitHub issue `#314` ("Moonshine streaming stall"). Not a
   Moonshine bug.

## Intermittency

The bug was intermittent. On 2026-05-15 roughly 1 in 5 cold boots produced
a working pipeline; the rest stalled. Kernel/driver state immediately
after a fresh boot appears to differ from steady-state behaviour. Once
the system settles into "MMAP-attenuates" mode, it stays there.

## The fix

[`audio_input/alsa_rw_capture.py`](../../src/robot_comic/audio_input/alsa_rw_capture.py)
spawns `arecord` (no `-M` flag, so default RW mode) and reads its stdout.
It feeds `LocalStream.record_loop` directly through a flag-gated source
indirection in [`console.py`](../../src/robot_comic/console.py). The
daemon's MMAP capture and TTS playback paths are untouched; `dsnoop`
tolerates concurrent readers.

The `REACHY_MINI_AUDIO_CAPTURE_PATH` env var (default `alsa_rw` on Linux,
`daemon` elsewhere) selects the path. On non-Linux hosts `AlsaRwCapture`
raises at `start()` so the daemon path is the only viable choice there.

## Reproducing the bug

After a daemon restart, or on a fresh boot when the bug fires:

```bash
# RW — full signal
arecord -D reachymini_audio_src -f S16_LE -r 16000 -c 2 -d 3 /tmp/rw.wav

# MMAP — ~1/10 signal
arecord -D reachymini_audio_src -M -f S16_LE -r 16000 -c 2 -d 3 /tmp/mmap.wav

# Compare peak/RMS:
python3 -c '
import wave, struct
for path in ["/tmp/rw.wav", "/tmp/mmap.wav"]:
    w = wave.open(path)
    n = w.getnframes(); ch = w.getnchannels()
    s = struct.unpack("<" + str(n*ch) + "h", w.readframes(n))
    peak = max(abs(x) for x in s)
    rms = int((sum(x*x for x in s)/(n*ch))**0.5)
    print(f"{path}: peak={peak} ({peak/32767:.3f}) rms={rms} ({rms/32767:.4f})")
'
```

Expect RW peak ≈ 1.0 and MMAP peak ≈ 0.1 during normal room sound.

[`scripts/dsnoop_multireader_check.py`](../../scripts/dsnoop_multireader_check.py)
ships in-repo for the same purpose: it exercises `AlsaRwCapture` against
the live device and prints peak/RMS over a 3-second window.

## Retest cadence

After every Reachy Mini daemon update, re-run the snippet above. If MMAP
and RW return the same levels (both 1.0 or both 0.1) the bug pattern has
shifted and the workaround should be revisited.

## Alternatives considered

- **Patch GStreamer's `alsasrc`** — out of scope; we don't ship a modified
  GStreamer.
- **In-process `pyalsaaudio`** — cleaner long-term, but adds a Linux-only
  Python dep and a cross-platform import-gate. The subprocess approach
  has zero new deps and the public API of `AlsaRwCapture` is small enough
  that a swap to `pyalsaaudio` later wouldn't touch consumers.
- **Restart-until-working** — last-resort workaround. Roughly 20% of
  boots work; bad UX.
