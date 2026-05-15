#!/usr/bin/env python3
"""Sanity-check: open AlsaRwCapture while the daemon is running.

Confirms that arecord -M no on reachymini_audio_src can coexist with the
daemon's MMAP-mode GStreamer reader.  Run on the Pi as Step 1 of the
field test:

    sudo systemctl is-active reachy-mini-daemon.service
    /venvs/apps_venv/bin/python scripts/dsnoop_multireader_check.py

Prints peak/RMS over a 3-second window.  Expect peak >= ~0.15 (with the
operator clapping or speaking in the room) — full-scale signal is 1.0.
Anything lower means the MMAP-attenuation bug has somehow propagated
to RW mode and the workaround needs revisiting.  See
docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md
and reference_alsa_mmap_attenuation (auto-memory) for context.
"""

from __future__ import annotations
import sys
import math
import time
import logging
import argparse

import numpy as np

from robot_comic.audio_input import AlsaRwCapture


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("dsnoop_multireader_check")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="reachymini_audio_src", help="ALSA device alias")
    parser.add_argument("--duration", type=float, default=3.0, help="Capture window seconds")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate Hz")
    parser.add_argument("--channels", type=int, default=2, help="Channel count")
    args = parser.parse_args(argv)

    cap = AlsaRwCapture(
        device=args.device,
        sample_rate=args.rate,
        channels=args.channels,
        frame_samples=256,
    )
    cap.start()
    logger.info("Capturing %.1fs from %s while daemon (presumably) runs ...", args.duration, args.device)

    deadline = time.monotonic() + args.duration
    collected: list[np.ndarray] = []
    while time.monotonic() < deadline:
        frame = cap.get_audio_sample()
        if frame is None:
            time.sleep(0.005)
            continue
        collected.append(frame.copy())

    cap.stop()

    if not collected:
        logger.error("No frames received — arecord likely failed.  Check stderr above.")
        return 1

    stacked = np.concatenate(collected, axis=0).astype(np.int32)
    peak = int(np.max(np.abs(stacked)))
    rms = int(math.sqrt(float(np.mean(stacked.astype(np.int64) ** 2))))
    peak_norm = peak / 32767.0
    rms_norm = rms / 32767.0

    logger.info(
        "Frames=%d samples=%d peak=%d (%.4f) rms=%d (%.4f)",
        len(collected),
        stacked.shape[0],
        peak,
        peak_norm,
        rms,
        rms_norm,
    )

    if peak_norm < 0.15:
        logger.warning(
            "Peak %.4f is below the 0.15 threshold — this looks like the MMAP-attenuation "
            "pattern leaking into RW mode.  Re-verify with `arecord -D %s -f S16_LE -d 3 /tmp/rw.wav` "
            "and a baseline `arecord -D %s -M -f S16_LE -d 3 /tmp/mmap.wav` per the memory.",
            peak_norm,
            args.device,
            args.device,
        )
        return 2
    logger.info("PASS — full-level signal received while daemon is running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
