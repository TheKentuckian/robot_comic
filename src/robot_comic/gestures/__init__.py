"""Gesture library — pre-canned movement sequences for Reachy Mini personas.

Usage
-----
Import the shared registry and play a gesture by name::

    from robot_comic.gestures import registry
    registry.play("shrug", movement_manager)

Or register custom gestures::

    from robot_comic.gestures import registry
    registry.register("my_gesture", my_fn)

The canonical gestures (shrug, nod_yes, nod_no, point_left, point_right,
scan, lean_in) are registered automatically when this package is imported.
"""

from robot_comic.gestures.registry import GestureRegistry


# Shared singleton registry — import and register canonical gestures.
registry = GestureRegistry()

# Register the built-in gesture set by importing the gestures module.
# This causes all `register_*` calls inside gestures.py to run.
from robot_comic.gestures import gestures as _gestures_module  # noqa: E402, F401


_gestures_module.register_all(registry)

__all__ = ["GestureRegistry", "registry"]
