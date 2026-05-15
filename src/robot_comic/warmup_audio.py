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
import wave
import shutil
import struct
import logging
import tempfile
import threading
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
    """Return the default ``<repo_root>/assets/welcome/welcome.wav`` path.

    This is the legacy single-file fallback. The split intro+picker pair
    lives next to it as ``welcome_intro.wav`` and ``welcome_picker.wav``;
    see :func:`default_welcome_intro_path` and
    :func:`default_welcome_picker_path`.
    """
    # src/robot_comic/warmup_audio.py -> src/robot_comic -> src -> repo root
    return Path(__file__).resolve().parents[2] / "assets" / "welcome" / "welcome.wav"


def default_welcome_intro_path() -> Path:
    """Return the default ``<repo_root>/assets/welcome/welcome_intro.wav`` path."""
    return Path(__file__).resolve().parents[2] / "assets" / "welcome" / "welcome_intro.wav"


def default_welcome_picker_path() -> Path:
    """Return the default ``<repo_root>/assets/welcome/welcome_picker.wav`` path."""
    return Path(__file__).resolve().parents[2] / "assets" / "welcome" / "welcome_picker.wav"


def _is_persona_locked() -> bool:
    """Return True when a custom/locked persona is active and the picker prompt
    should be suppressed.

    Reads from :mod:`robot_comic.config` so that both the build-time
    ``LOCKED_PROFILE`` constant and the runtime ``REACHY_MINI_CUSTOM_PROFILE``
    env var are honoured (the config module merges them into the same
    attribute at startup). Falls back to a direct env-var read if config has
    not been imported yet — warmup fires very early in boot.
    """
    try:
        from robot_comic import config  # noqa: PLC0415 — late import to avoid boot cost

        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if profile:
            return True
    except Exception:
        # config import can fail very early in boot; fall through to env check
        pass
    return bool(os.environ.get("REACHY_MINI_CUSTOM_PROFILE", "").strip())


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


def _dispatch_single_wav(wav_path: Path) -> bool:
    """Fire-and-forget play of a single WAV. Returns True if dispatched.

    Mirrors the per-platform branching used by :func:`play_warmup_wav`, but
    factored out so callers (and the chained intro+picker path) can dispatch
    one file at a time without re-implementing player detection.
    """
    global _PLAYER_WARNED

    if not wav_path.is_file():
        logger.info("Warmup WAV not found at %s; skipping", wav_path)
        return False

    if sys.platform == "win32":
        dispatched = _play_windows(wav_path)
        if dispatched:
            logger.info("Warmup WAV dispatched (winsound): %s", wav_path.name)
        return dispatched

    if _PLAYER_CMD is None:
        if not _PLAYER_WARNED:
            logger.warning("No audio player available (looked for aplay/paplay/afplay); skipping warmup WAV")
            _PLAYER_WARNED = True
        return False

    cmd = [*_PLAYER_CMD, str(wav_path)]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        logger.info("Warmup WAV dispatched: %s", wav_path.name)
        return True
    except Exception as exc:
        logger.warning("Failed to spawn warmup player %s: %s", cmd[0], exc)
        return False


def play_warmup_wav(path: str | Path | None = None) -> None:
    """Dispatch warmup audio playback and return immediately.

    Behaviour matrix:
    - When ``path`` is provided explicitly: play that single file (legacy/test
      callers, ``ROBOT_COMIC_WARMUP_WAV`` env override).
    - When ``path`` is None and the split assets exist
      (``welcome_intro.wav`` + ``welcome_picker.wav``):
      * If a persona is locked (``REACHY_MINI_CUSTOM_PROFILE`` /
        ``LOCKED_PROFILE``): play only the intro — the picker prompt
        ("pick your comedian") would be misleading.
      * Otherwise: play intro, then chain the picker on a daemon thread.
    - When the split assets are missing: fall back to legacy
      ``welcome.wav``.

    Silent no-op if no player is on PATH. Never raises; warmup is a
    best-effort UX nicety and must not block startup.

    Emits a ``warmup wav dispatched`` startup checkpoint (via
    ``startup_timer.log_checkpoint``) only when at least one subprocess /
    winsound call is actually spawned, so the ``+Xs warmup wav dispatched →
    first TTS audio frame`` delta is meaningful. Skipped paths emit
    ``warmup wav skipped`` at INFO so the journal always records which path
    was taken.

    Optional blip
    ~~~~~~~~~~~~~
    When ``REACHY_MINI_WARMUP_BLIP_ENABLED=1`` is set, a tiny synthesised
    tone (:func:`play_warmup_blip`) is played first for instant "alive"
    feedback, then the welcome WAV(s) follow as usual.
    """
    from robot_comic.startup_timer import log_checkpoint

    # ``main.py`` dispatches the welcome WAV before any non-stdlib import so
    # the operator hears it within ~1s of ``systemctl start`` (vs ~5-15s for
    # this in-process path). When that early path fires it sets the env flag
    # below; skip cleanly so we don't double-play.
    if os.environ.get("REACHY_MINI_EARLY_WELCOME_PLAYED") == "1":
        logger.info("Warmup WAV already played by early-dispatch in main.py; skipping")
        log_checkpoint("warmup wav skipped", logger)
        return

    # --- optional fast-blip path -------------------------------------------
    blip_enabled = os.getenv("REACHY_MINI_WARMUP_BLIP_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if blip_enabled:
        play_warmup_blip()

    # --- explicit path: legacy single-file behaviour -----------------------
    if path is not None:
        wav_path = Path(path)
        dispatched = _dispatch_single_wav(wav_path)
        log_checkpoint("warmup wav dispatched" if dispatched else "warmup wav skipped", logger)
        return

    # --- default: prefer split intro+picker; fall back to legacy welcome.wav
    intro_path = default_welcome_intro_path()
    picker_path = default_welcome_picker_path()

    if intro_path.is_file() and picker_path.is_file():
        intro_dispatched = _dispatch_single_wav(intro_path)
        if not intro_dispatched:
            log_checkpoint("warmup wav skipped", logger)
            return

        if _is_persona_locked():
            logger.info("Persona locked; skipping welcome picker prompt")
            log_checkpoint("warmup wav dispatched", logger)
            return

        # Chain picker after intro on a daemon thread so we don't block boot.
        t = threading.Thread(
            target=_chain_intro_then_picker_after_dispatch,
            args=(intro_path, picker_path),
            name="warmup-picker-chain",
            daemon=True,
        )
        t.start()
        log_checkpoint("warmup wav dispatched", logger)
        return

    # Legacy fallback: either split file missing — play the old single welcome.wav
    legacy_path = default_warmup_wav_path()
    logger.info(
        "Split welcome assets missing (intro=%s, picker=%s); falling back to %s",
        intro_path.is_file(),
        picker_path.is_file(),
        legacy_path.name,
    )
    dispatched = _dispatch_single_wav(legacy_path)
    log_checkpoint("warmup wav dispatched" if dispatched else "warmup wav skipped", logger)


def _chain_intro_then_picker_after_dispatch(intro: Path, picker: Path) -> None:
    """Worker-thread target: wait for the already-dispatched intro to finish,
    then play the picker.

    The intro was already dispatched fire-and-forget by the caller (so the
    checkpoint fires on the boot path), so here we just re-play the intro
    *blockingly* to wait out its duration before triggering the picker.
    A second simultaneous open of the same WAV via the shared player is
    benign — pw-play / paplay route through the daemon's mix sink, aplay
    fails silently when the device is busy, and winsound serialises.

    To avoid the doubled audio we instead use a duration-based sleep derived
    from the WAV header, which is cheap and platform-agnostic.
    """
    try:
        with wave.open(str(intro), "rb") as wf:
            duration_s = wf.getnframes() / float(wf.getframerate())
    except Exception as exc:
        logger.warning("Could not read intro duration from %s: %s", intro.name, exc)
        duration_s = 0.0

    # Small safety pad so the picker doesn't clip onto the intro tail.
    import time  # noqa: PLC0415 — narrow scope, daemon-thread only

    time.sleep(max(0.0, duration_s) + 0.05)
    _dispatch_single_wav(picker)
