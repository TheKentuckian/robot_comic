"""Fire-and-forget warmup WAV playback for headless on-robot startup.

Plays a short pre-rendered greeting through the on-robot speakers as soon as
audio output is possible, in parallel with the rest of the stack loading. The
real handler will take over the audio device once it starts driving the
speakers; the warmup subprocess being cut off mid-playback is expected.

Sim mode never invokes this — there are no on-robot speakers and FastRTC is
not up yet.
"""

from __future__ import annotations
import sys
import shutil
import logging
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)


def _detect_player() -> list[str] | None:
    """Resolve the player command + flags, or None if unavailable.

    Prefers ALSA's ``aplay``, falls back to PulseAudio's ``paplay``. On macOS,
    ``afplay`` is used if available. Windows has no zero-dep CLI player, so
    this returns None there.
    """
    if sys.platform == "win32":
        return None
    if sys.platform == "darwin":
        afplay = shutil.which("afplay")
        if afplay:
            return [afplay]
        return None
    aplay = shutil.which("aplay")
    if aplay:
        return [aplay, "-q"]
    paplay = shutil.which("paplay")
    if paplay:
        return [paplay]
    return None


_PLAYER_CMD: list[str] | None = _detect_player()
_PLAYER_WARNED = False


def default_warmup_wav_path() -> Path:
    """Return the default ``<repo_root>/assets/welcome/welcome.wav`` path."""
    # src/robot_comic/warmup_audio.py -> src/robot_comic -> src -> repo root
    return Path(__file__).resolve().parents[2] / "assets" / "welcome" / "welcome.wav"


def play_warmup_wav(path: str | Path | None = None) -> None:
    """Spawn a player subprocess for the warmup WAV and return immediately.

    Silent no-op if the file does not exist or no player is on PATH. Never
    raises; warmup is a best-effort UX nicety and must not block startup.

    Emits a ``warmup wav dispatched`` startup checkpoint (via
    ``startup_timer.log_checkpoint``) only when a subprocess is actually
    spawned, so the ``+Xs warmup wav dispatched → first TTS audio frame``
    delta is meaningful. Skipped paths emit ``warmup wav skipped`` at INFO
    so the journal always records which path was taken.
    """
    global _PLAYER_WARNED

    from robot_comic.startup_timer import log_checkpoint

    wav_path = Path(path) if path else default_warmup_wav_path()
    if not wav_path.is_file():
        logger.info("Warmup WAV not found at %s; skipping", wav_path)
        log_checkpoint("warmup wav skipped", logger)
        return

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
    except Exception as exc:
        logger.warning("Failed to spawn warmup player %s: %s", cmd[0], exc)
        log_checkpoint("warmup wav skipped", logger)
