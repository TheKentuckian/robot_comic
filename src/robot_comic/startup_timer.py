"""Shared startup stopwatch.

Imported as early as possible so ``STARTUP_T0`` captures the earliest Python
time we can measure. All ``Startup: +X.XXs <label>`` checkpoints share this
origin so logs from main and from handler-side first-event hooks are
comparable.
"""

from __future__ import annotations
import time
import logging


STARTUP_T0: float = time.perf_counter()

_logger = logging.getLogger(__name__)


def since_startup() -> float:
    """Seconds elapsed since module import."""
    return time.perf_counter() - STARTUP_T0


def log_checkpoint(label: str, logger: logging.Logger | None = None) -> None:
    """Emit a ``Startup: +X.XXs <label>`` line at INFO level."""
    (logger or _logger).info("Startup: +%.2fs %s", since_startup(), label)


_FIRED: set[str] = set()


def log_once(label: str, logger: logging.Logger | None = None) -> None:
    """Emit a startup checkpoint exactly once per process, keyed by label.

    Used for first-event hooks scattered across handler subclasses (first LLM
    token, first TTS audio frame) where a class-level guard would need to be
    duplicated across multiple files.
    """
    if label in _FIRED:
        return
    _FIRED.add(label)
    log_checkpoint(label, logger)
