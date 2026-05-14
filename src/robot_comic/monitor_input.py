r"""Non-blocking keyboard input helper for the robot-comic TUI monitor.

Provides a single cross-platform entry point, ``MonitorInput``, whose
``poll_key`` method returns a single character (or ``None``) without blocking
the render loop.

Platform strategies
-------------------
POSIX (Linux / macOS):
    Switch stdin to *cbreak* mode via :mod:`termios` so keystrokes are
    delivered one character at a time without waiting for Enter.
    :func:`select.select` is then used with a short timeout to avoid blocking.

Windows:
    :mod:`msvcrt` provides ``kbhit()``/``getch()`` which are inherently
    non-blocking.

A small number of raw byte sequences are normalised to logical names:

* ``\x1b`` (ESC alone)  -> ``"<esc>"``
* ``\r`` / ``\n`` (Enter) -> ``"<enter>"``
* ``\x03`` (Ctrl-C)    -> ``"<interrupt>"``

All other printable characters are returned as-is (lower-cased for
letters so callers can match ``'s'`` regardless of Caps Lock).
"""

from __future__ import annotations
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Internal platform detection
# ---------------------------------------------------------------------------

_WINDOWS = os.name == "nt"


def _make_backend() -> "_InputBackend":
    if _WINDOWS:
        return _WindowsBackend()
    return _PosixBackend()


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

_BYTE_MAP: dict[bytes, str] = {
    b"\x1b": "<esc>",
    b"\r": "<enter>",
    b"\n": "<enter>",
    b"\x03": "<interrupt>",
}


def _normalise(raw: bytes) -> Optional[str]:
    """Convert a raw byte(s) read from stdin to a logical key name.

    Returns ``None`` for non-printable / control sequences we don't handle.
    """
    if raw in _BYTE_MAP:
        return _BYTE_MAP[raw]
    if len(raw) == 1 and 0x20 <= raw[0] < 0x7F:
        return chr(raw[0]).lower()
    return None


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


class _InputBackend:
    """Abstract non-blocking stdin backend."""

    def read_char(self, timeout: float) -> Optional[bytes]:
        """Return one raw byte within *timeout* seconds, or ``None``."""
        raise NotImplementedError  # pragma: no cover

    def close(self) -> None:
        """Restore any terminal state changed during construction."""


class _PosixBackend(_InputBackend):
    """POSIX backend using termios + select."""

    def __init__(self) -> None:
        """Switch stdin to cbreak (single-char) mode."""
        try:
            import tty
            import termios

            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)  # type: ignore[attr-defined]
            tty.cbreak(self._fd)  # type: ignore[attr-defined]
        except Exception:
            # stdin is not a tty (e.g. redirected in tests) — fall back to a
            # no-op backend so callers don't crash.
            self._fd = -1
            self._old_settings = None

    def read_char(self, timeout: float) -> Optional[bytes]:
        """Non-blocking single-char read with *timeout* second wait."""
        if self._fd < 0:
            return None
        try:
            import select

            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if r:
                return sys.stdin.buffer.read(1)
        except Exception:
            pass
        return None

    def close(self) -> None:
        """Restore the terminal to its original settings."""
        if self._fd >= 0 and self._old_settings is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)  # type: ignore[attr-defined]
            except Exception:
                pass


class _WindowsBackend(_InputBackend):
    """Windows backend using msvcrt."""

    def read_char(self, timeout: float) -> Optional[bytes]:
        """Poll msvcrt and return a byte if one is waiting, else wait up to *timeout*."""
        import time
        import msvcrt

        deadline = time.perf_counter() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                # Arrow keys / function keys produce a two-byte sequence starting
                # with 0x00 or 0xE0 — consume and discard the second byte.
                if ch in (b"\x00", b"\xe0"):
                    msvcrt.getch()
                    return None
                return ch
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            time.sleep(min(0.02, remaining))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MonitorInput:
    """Non-blocking keyboard poller for the monitor render loop.

    Usage::

        inp = MonitorInput()
        try:
            key = inp.poll_key(timeout=0.1)  # returns str | None
        finally:
            inp.close()

    Thread-safety: ``poll_key`` is **not** thread-safe; call it from a single
    thread only (the monitor's main loop is the intended caller).
    """

    def __init__(self) -> None:
        """Initialise the platform backend."""
        self._backend = _make_backend()

    def poll_key(self, timeout: float = 0.1) -> Optional[str]:
        """Return the next key pressed within *timeout* seconds, or ``None``.

        The return value is a logical key name:
        * Printable ASCII → lower-cased character string
        * Enter           → ``"<enter>"``
        * Escape          → ``"<esc>"``
        * Ctrl-C          → ``"<interrupt>"``
        * Everything else → ``None`` (silently ignored)
        """
        raw = self._backend.read_char(timeout)
        if raw is None:
            return None
        return _normalise(raw)

    def close(self) -> None:
        """Restore terminal state. Call when the monitor exits."""
        self._backend.close()
