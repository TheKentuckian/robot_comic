"""Direct ALSA audio capture, bypassing the daemon's MMAP-attenuated path.

See docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md.
"""

from robot_comic.audio_input.alsa_rw_capture import AlsaRwCapture


__all__ = ["AlsaRwCapture"]
