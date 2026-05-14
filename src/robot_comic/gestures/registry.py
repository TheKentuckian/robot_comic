"""GestureRegistry — maps gesture names to callable movement functions.

A gesture function has the signature::

    def my_gesture(manager: MovementManager) -> None: ...

It is expected to call ``manager.queue_move(...)`` with one or more
``Move`` instances.  Gestures are primary moves, so they run exclusively
and sequentially via the existing ``MovementManager`` queue.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Callable


if TYPE_CHECKING:
    from robot_comic.moves import MovementManager


logger = logging.getLogger(__name__)

GestureFn = Callable[["MovementManager"], None]


class GestureRegistry:
    """Registry of named gesture functions.

    Thread-safety note: ``register`` and ``list_gestures`` are typically
    called at startup/import time.  ``play`` is called from tool dispatch
    threads but only reads from the registry dict after registration is
    complete, so no locking is required for the current use-pattern.
    """

    def __init__(self) -> None:
        """Initialise an empty gesture registry."""
        self._gestures: Dict[str, GestureFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, fn: GestureFn) -> None:
        """Register a gesture function under *name*.

        Parameters
        ----------
        name:
            Canonical snake_case identifier (e.g. ``"shrug"``).
        fn:
            Callable that accepts a ``MovementManager`` and enqueues
            primary moves on it.

        """
        if name in self._gestures:
            logger.warning("GestureRegistry: overwriting existing gesture %r", name)
        self._gestures[name] = fn
        logger.debug("GestureRegistry: registered gesture %r", name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_gestures(self) -> List[str]:
        """Return a sorted list of all registered gesture names."""
        return sorted(self._gestures)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play(self, name: str, manager: "MovementManager") -> None:
        """Queue the named gesture's moves on *manager*.

        Parameters
        ----------
        name:
            Gesture name as returned by :meth:`list_gestures`.
        manager:
            The live ``MovementManager`` instance.

        Raises
        ------
        KeyError
            If *name* is not registered.  The message lists available
            gestures so callers can surface a helpful error to the LLM.

        """
        fn = self._gestures.get(name)
        if fn is None:
            available = ", ".join(self.list_gestures()) or "<none>"
            raise KeyError(f"Unknown gesture {name!r}. Available gestures: {available}")
        logger.info("GestureRegistry: playing gesture %r", name)
        fn(manager)
