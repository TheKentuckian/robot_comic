"""arecord-subprocess RW-mode capture from reachymini_audio_src.

Why this exists: the dsnoop device reachymini_audio_src delivers ~1/10 the
signal level under MMAP-interleaved access vs RW-interleaved access on this
hardware.  GStreamer's alsasrc defaults to MMAP with no override knob, so
the daemon's audio capture pipeline produces near-silent audio that
Moonshine correctly reports as "no speech."  By spawning arecord in RW
mode ourselves and feeding LocalStream.record_loop directly, we get the
full-level signal without touching the daemon.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


class AlsaRwCapture:
    """Read S16_LE frames from an ALSA device in RW-interleaved mode."""

    def __init__(
        self,
        device: str = "reachymini_audio_src",
        sample_rate: int = 16000,
        channels: int = 2,
        frame_samples: int = 256,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_samples = frame_samples
        self._frame_bytes = frame_samples * channels * 2  # S16 = 2 bytes/sample
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._buf = bytearray()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        if sys.platform != "linux":
            raise RuntimeError(
                f"AlsaRwCapture is Linux-only (current platform: {sys.platform}). "
                "Use AUDIO_CAPTURE_PATH=daemon on non-Linux hosts."
            )
        if self._proc is not None:
            raise RuntimeError("AlsaRwCapture already started")
        cmd = [
            "arecord",
            "-q",
            "-D",
            self._device,
            "-M",
            "no",
            "-f",
            "S16_LE",
            "-r",
            str(self._sample_rate),
            "-c",
            str(self._channels),
            "-t",
            "raw",
        ]
        logger.info("Starting AlsaRwCapture: %s", " ".join(cmd))
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def get_audio_sample(self) -> Optional[np.ndarray[Any, Any]]:
        """Return next (frame_samples, channels) int16 frame, or None if no full frame ready."""
        if self._proc is None or self._proc.stdout is None:
            return None
        needed = self._frame_bytes - len(self._buf)
        if needed > 0:
            chunk = self._proc.stdout.read(needed)
            if chunk:
                self._buf.extend(chunk)
        if len(self._buf) < self._frame_bytes:
            return None
        frame_bytes = bytes(self._buf[: self._frame_bytes])
        del self._buf[: self._frame_bytes]
        return np.frombuffer(frame_bytes, dtype=np.int16).reshape(self._frame_samples, self._channels)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.warning("arecord did not exit on SIGTERM, sending SIGKILL")
                proc.kill()
                proc.wait(timeout=1.0)
        except Exception as e:
            logger.warning("Error stopping arecord: %s", e)
        finally:
            # Drain stderr for diagnostics.
            try:
                if proc.stderr is not None:
                    err = proc.stderr.read()
                    if err:
                        logger.info("arecord stderr at shutdown: %s", err.decode(errors="replace").strip())
            except Exception:
                pass
        self._buf.clear()
