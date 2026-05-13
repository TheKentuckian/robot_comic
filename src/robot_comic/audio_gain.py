"""Post-synthesis gain normalisation for Chatterbox TTS output.

Measures the RMS of a synthesised PCM clip and scales it so the output
reaches a configurable target loudness (default −16 dBFS).  All arithmetic
stays in float32; the final result is clipped to the int16 range before
being returned as int16.

Public surface
--------------
normalize_gain(pcm, target_dbfs=-16.0) -> np.ndarray
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

# Clips whose RMS is below this threshold (≈ −80 dBFS) are treated as silence
# and returned unchanged so we don't amplify noise floors to absurd levels.
_SILENCE_FLOOR_RMS: float = 10.0 ** (-80.0 / 20.0) * 32768.0  # ~3.3 counts

_INT16_MAX: float = 32767.0
_INT16_MIN: float = -32768.0


def normalize_gain(
    pcm: "np.ndarray[Any, np.dtype[np.int16]]",
    target_dbfs: float = -16.0,
) -> "np.ndarray[Any, np.dtype[np.int16]]":
    """Scale *pcm* so its RMS reaches *target_dbfs* dBFS.

    Parameters
    ----------
    pcm:
        1-D int16 numpy array (mono PCM).  Must already be mono; stereo is
        not supported here (conversion happens upstream in ``_wav_to_pcm``).
    target_dbfs:
        Target RMS level in dBFS.  Must be ≤ 0.  Typical values:
        −16 dBFS (default, broadcast-safe),  −18 dBFS (slightly quieter).

    Returns
    -------
    np.ndarray[np.int16]
        Gain-adjusted PCM clipped to the int16 range.  Returns the *original*
        array unchanged when the input is silence (RMS below the floor).

    """
    if pcm.size == 0:
        return pcm

    audio = pcm.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio**2)))

    if rms < _SILENCE_FLOOR_RMS:
        # Treat as silence — do not amplify noise.
        return pcm

    # Target RMS in linear int16 scale.
    # target_dbfs is relative to full-scale (32768 for int16).
    target_rms = (10.0 ** (target_dbfs / 20.0)) * 32768.0
    scale = target_rms / rms

    scaled = audio * scale

    # Detect hard clipping before committing.
    if np.any(scaled > _INT16_MAX) or np.any(scaled < _INT16_MIN):
        logger.warning(
            "normalize_gain: hard clipping triggered (target_dbfs=%.1f, rms=%.1f, scale=%.3f). "
            "Consider lowering REACHY_MINI_CHATTERBOX_TARGET_DBFS.",
            target_dbfs,
            rms,
            scale,
        )

    clipped: np.ndarray[Any, np.dtype[np.int16]] = np.clip(scaled, _INT16_MIN, _INT16_MAX).astype(np.int16)
    return clipped
