"""Fire-and-forget warmup WAV playback for headless on-robot startup.

Plays a short pre-rendered greeting through the on-robot speakers as soon as
audio output is possible, in parallel with the rest of the stack loading. The
real handler will take over the audio device once it starts driving the
speakers; the warmup subprocess being cut off mid-playback is expected.

Sim mode never invokes this — there are no on-robot speakers and FastRTC is
not up yet.

Platform support
----------------
- Linux: ``pw-play`` (PipeWire) → ``paplay`` (PulseAudio) → ``aplay`` (ALSA)
- macOS: ``afplay``
- Windows: ``winsound.PlaySound`` with ``SND_ASYNC`` (stdlib, no extra deps)

Optional fast-blip path
-----------------------
When ``REACHY_MINI_WARMUP_BLIP_ENABLED=1`` is set, a tiny synthesised WAV
(200 ms sine tone) is played *first* via the same player pipeline. This is
generated entirely in-process so it has no file-I/O dependency, giving
immediate audio feedback even before the welcome WAV has been read from disk.
The blip is generated with the standard library ``wave`` module + ``struct``
(no NumPy dependency). Set ``REACHY_MINI_WARMUP_BLIP_ENABLED=0`` or leave
unset to keep the previous behaviour (blip disabled).
"""

from __future__ import annotations
import io
import os
import sys
import math
import time
import wave
import shutil
import struct
import logging
import tempfile
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Player detection
# ---------------------------------------------------------------------------


def _detect_player() -> list[str] | None:
    """Resolve the player command + flags, or None if unavailable.

    Prefers PipeWire's ``pw-play`` then PulseAudio's ``paplay`` (both share
    the audio device with the reachy_mini daemon), falling back to ALSA's
    ``aplay`` (exclusive — silently fails when the daemon owns the device).
    On macOS, ``afplay`` is used if available. On Windows, ``winsound`` is
    used directly (see ``_play_windows``); this function returns None there so
    the caller can dispatch to the Windows path instead.
    """
    if sys.platform == "win32":
        return None
    if sys.platform == "darwin":
        afplay = shutil.which("afplay")
        if afplay:
            return [afplay]
        return None
    pw_play = shutil.which("pw-play")
    if pw_play:
        return [pw_play]
    paplay = shutil.which("paplay")
    if paplay:
        return [paplay]
    aplay = shutil.which("aplay")
    if aplay:
        cmd = [aplay, "-q"]
        # When the reachy_mini daemon holds the USB speaker (`/dev/snd/pcmC0D0p`)
        # exclusively via mmap, the default ALSA hw device is busy and `aplay`
        # fails silently with "Device or resource busy". Reachy Mini ships an
        # `~/.asoundrc` that defines a `dmix`-backed `reachymini_audio_sink`
        # which allows concurrent openers; on-robot deployments should set
        # `REACHY_MINI_ALSA_DEVICE=plug:reachymini_audio_sink` to route through
        # it (the `plug:` prefix handles format/rate conversion automatically).
        # On hosts without that PCM defined (CI / dev laptops), leave the env
        # var unset and aplay uses the default device.
        device = os.environ.get("REACHY_MINI_ALSA_DEVICE", "").strip()
        if device:
            cmd.extend(["-D", device])
        return cmd
    return None


_PLAYER_CMD: list[str] | None = _detect_player()
_PLAYER_WARNED = False

# ---------------------------------------------------------------------------
# Blip generator
# ---------------------------------------------------------------------------

_BLIP_SAMPLE_RATE = 22050
_BLIP_DURATION_S = 0.20  # seconds
_BLIP_FREQ_HZ = 880.0  # A5 — audible, not harsh
_BLIP_AMPLITUDE = 0.35  # 0-1; keep well below clipping


def generate_blip_wav_bytes() -> bytes:
    """Generate a short sine-tone WAV in memory.

    Returns raw WAV bytes for a :data:`_BLIP_DURATION_S` second sine wave at
    :data:`_BLIP_FREQ_HZ` Hz, :data:`_BLIP_SAMPLE_RATE` Hz sample rate, mono,
    16-bit signed PCM. No NumPy or other third-party dependency is required.
    """
    n_samples = int(_BLIP_SAMPLE_RATE * _BLIP_DURATION_S)
    peak = int(_BLIP_AMPLITUDE * 32767)

    # Apply a simple linear fade-in / fade-out to avoid clicks
    fade_samples = min(int(_BLIP_SAMPLE_RATE * 0.01), n_samples // 4)  # 10 ms

    raw_samples: list[int] = []
    for i in range(n_samples):
        t = i / _BLIP_SAMPLE_RATE
        sample = peak * math.sin(2 * math.pi * _BLIP_FREQ_HZ * t)
        # Fade envelope
        if i < fade_samples:
            sample *= i / fade_samples
        elif i >= n_samples - fade_samples:
            sample *= (n_samples - i) / fade_samples
        raw_samples.append(int(sample))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(_BLIP_SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{n_samples}h", *raw_samples))

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Windows playback helper
# ---------------------------------------------------------------------------


def _play_windows(wav_path: Path) -> bool:
    """Play *wav_path* asynchronously using ``winsound``.

    Returns True when playback was dispatched, False on error. The
    ``SND_ASYNC`` flag causes ``PlaySound`` to return immediately, so startup
    is not blocked. ``SND_NODEFAULT`` suppresses the system default beep if
    the file cannot be found (belt-and-suspenders guard since we already check
    ``wav_path.is_file()`` before calling this).
    """
    try:
        import winsound  # type: ignore[import-not-found,unused-ignore] # noqa: PLC0415 — Windows-only stdlib module

        flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT  # type: ignore[attr-defined,unused-ignore]
        winsound.PlaySound(str(wav_path), flags)  # type: ignore[attr-defined,unused-ignore]
        return True
    except Exception as exc:
        logger.warning("winsound.PlaySound failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def default_warmup_wav_path() -> Path:
    """Return the default ``<repo_root>/assets/welcome/welcome.wav`` path."""
    # src/robot_comic/warmup_audio.py -> src/robot_comic -> src -> repo root
    return Path(__file__).resolve().parents[2] / "assets" / "welcome" / "welcome.wav"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def play_warmup_blip() -> bool:
    """Play a tiny synthesised audio cue immediately, before any file I/O.

    Generates a :data:`_BLIP_DURATION_S` second sine tone in memory and plays
    it through the same player pipeline as :func:`play_warmup_wav`. On
    Windows, ``winsound`` is used with the ``SND_ASYNC`` flag. The blip gives
    instant "alive" feedback regardless of whether the welcome WAV file exists
    on disk.

    Returns True when playback was dispatched, False when no player is
    available or an error occurred. Never raises.
    """
    try:
        wav_bytes = generate_blip_wav_bytes()
    except Exception as exc:
        logger.warning("Blip generation failed: %s", exc)
        return False

    # Windows: write to a temp file and use winsound
    if sys.platform == "win32":
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = Path(tmp.name)
            dispatched = _play_windows(tmp_path)
            # The temp file may be locked until winsound finishes; schedule
            # removal via atexit so we don't leave stale files on disk.
            import atexit

            atexit.register(_safe_remove, tmp_path)
            return dispatched
        except Exception as exc:
            logger.warning("Blip temp-file write failed: %s", exc)
            return False

    # POSIX: pipe via subprocess stdin (pw-play / paplay / aplay all accept
    # stdin when given "-" or when the file argument is omitted for aplay).
    # For simplicity, write to a temp file to keep the subprocess invocation
    # identical to the WAV path.
    if _PLAYER_CMD is None:
        logger.debug("No audio player available; blip skipped")
        return False

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)
        cmd = [*_PLAYER_CMD, str(tmp_path)]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        import atexit

        atexit.register(_safe_remove, tmp_path)
        logger.debug("Warmup blip dispatched")
        return True
    except Exception as exc:
        logger.warning("Failed to spawn blip player: %s", exc)
        return False


def _safe_remove(path: Path) -> None:
    """Remove *path* silently; used as an atexit handler for temp blip files."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def play_warmup_wav(path: str | Path | None = None) -> None:
    """Spawn a player subprocess for the warmup WAV and return immediately.

    Silent no-op if the file does not exist or no player is on PATH. Never
    raises; warmup is a best-effort UX nicety and must not block startup.

    Emits a ``warmup wav dispatched`` startup checkpoint (via
    ``startup_timer.log_checkpoint``) only when a subprocess is actually
    spawned, so the ``+Xs warmup wav dispatched → first TTS audio frame``
    delta is meaningful. Skipped paths emit ``warmup wav skipped`` at INFO
    so the journal always records which path was taken.

    Optional blip
    ~~~~~~~~~~~~~
    When ``REACHY_MINI_WARMUP_BLIP_ENABLED=1`` is set, a tiny synthesised
    tone (:func:`play_warmup_blip`) is played first for instant "alive"
    feedback, then the full welcome WAV follows as usual.
    """
    global _PLAYER_WARNED

    from robot_comic import telemetry as _telemetry
    from robot_comic.startup_timer import log_checkpoint

    _wav_started_at = time.monotonic()

    def _emit_supporting(dur_s: float) -> None:
        """Surface ``welcome.wav.played`` on the monitor boot-timeline (#301).

        The aplay/winsound dispatch is fire-and-forget so we can only measure
        the wall-clock spent inside this function (subprocess spawn cost). It
        still gives the operator a single anchor for "the warmup WAV path
        executed at this time".
        """
        try:
            _telemetry.emit_supporting_event("welcome.wav.played", dur_ms=dur_s * 1000)
        except Exception:
            pass

    # --- optional fast-blip path -------------------------------------------
    blip_enabled = os.getenv("REACHY_MINI_WARMUP_BLIP_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if blip_enabled:
        play_warmup_blip()

    # --- welcome WAV path ---------------------------------------------------
    wav_path = Path(path) if path else default_warmup_wav_path()
    if not wav_path.is_file():
        logger.info("Warmup WAV not found at %s; skipping", wav_path)
        log_checkpoint("warmup wav skipped", logger)
        return

    # Windows: use winsound directly (no external player needed)
    if sys.platform == "win32":
        dispatched = _play_windows(wav_path)
        if dispatched:
            logger.info("Warmup WAV dispatched (winsound): %s", wav_path.name)
            log_checkpoint("warmup wav dispatched", logger)
            _emit_supporting(time.monotonic() - _wav_started_at)
        else:
            log_checkpoint("warmup wav skipped", logger)
        return

    # POSIX: spawn subprocess player
    if _PLAYER_CMD is None:
        if not _PLAYER_WARNED:
            logger.warning("No audio player available (looked for aplay/paplay/afplay); skipping warmup WAV")
            _PLAYER_WARNED = True
        log_checkpoint("warmup wav skipped", logger)
        return

    cmd = [*_PLAYER_CMD, str(wav_path)]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        logger.info("Warmup WAV dispatched: %s", wav_path.name)
        log_checkpoint("warmup wav dispatched", logger)
        _emit_supporting(time.monotonic() - _wav_started_at)
    except Exception as exc:
        logger.warning("Failed to spawn warmup player %s: %s", cmd[0], exc)
        log_checkpoint("warmup wav skipped", logger)
