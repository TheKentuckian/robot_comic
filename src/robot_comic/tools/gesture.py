"""Gesture tool — lets the LLM trigger pre-canned movement sequences.

Personas opt in by listing ``gesture`` in their ``tools.txt``.
"""

from __future__ import annotations
import logging
from typing import Any, Dict

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Lazily import the registry so the module can be loaded even in test
# environments that don't have the full reachy_mini SDK available.
def _get_registry() -> Any:
    from robot_comic.gestures import registry  # noqa: PLC0415

    return registry


class Gesture(Tool):
    """Play a named gesture (pre-canned movement sequence).

    Available gestures: shrug, nod_yes, nod_no, point_left, point_right,
    scan, lean_in.
    """

    name = "gesture"
    description = (
        "Play a pre-canned physical gesture. Use to add expressive body "
        "language that matches your delivery: shrug when uncertain, nod_yes "
        "to agree, nod_no to disagree, point_left / point_right to direct "
        "attention, scan to survey the audience, lean_in when sharing a "
        "punchline or aside."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": [
                    "shrug",
                    "nod_yes",
                    "nod_no",
                    "point_left",
                    "point_right",
                    "scan",
                    "lean_in",
                ],
                "description": (
                    "Gesture to perform. "
                    "shrug — uncertain / comic timing; "
                    "nod_yes — affirmative; "
                    "nod_no — negative / dismissive; "
                    "point_left / point_right — direct attention; "
                    "scan — slow theatrical sweep of the audience; "
                    "lean_in — conspiratorial aside or punchline build-up."
                ),
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play the requested gesture."""
        gesture_name = kwargs.get("name")
        if not isinstance(gesture_name, str) or not gesture_name:
            return {"error": "gesture name must be a non-empty string"}

        logger.info("Tool call: gesture name=%s", gesture_name)

        try:
            reg = _get_registry()
            reg.play(gesture_name, deps.movement_manager)
            return {"status": "queued", "gesture": gesture_name}
        except KeyError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("gesture tool failed: %s", exc)
            return {"error": f"gesture failed: {type(exc).__name__}: {exc}"}
