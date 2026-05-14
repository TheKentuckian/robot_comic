import random
import logging
from typing import Any, Dict

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Emotions to filter out at the tool layer. These animations have keyframe
# amplitudes large enough that on at least one chassis they swing the head
# into the surround cowling. Until per-emotion velocity clamping lands
# (see #264), drop them entirely.
BLOCKED_EMOTION_PREFIXES: tuple[str, ...] = ("lonely",)


def _is_blocked_emotion(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in BLOCKED_EMOTION_PREFIXES)


# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False


def _safe_emotion_names() -> list[str]:
    """Return the available emotion names with blocked prefixes filtered out."""
    if not EMOTION_AVAILABLE:
        return []
    return [n for n in RECORDED_MOVES.list_moves() if not _is_blocked_emotion(n)]


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = _safe_emotion_names()
        if not emotion_names:
            return "No emotions currently available"

        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": _safe_emotion_names(),
                "description": f"""Name of the emotion to play; omit for random.
                                    Here is a list of the available emotions, you MUST only choose from these: \n
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Defence in depth: if the LLM hallucinates a blocked emotion despite
        # it being absent from the enum, refuse here too.
        if emotion_name and _is_blocked_emotion(emotion_name):
            logger.warning("Refusing blocked emotion %r (matches BLOCKED_EMOTION_PREFIXES)", emotion_name)
            return {"error": f"Emotion '{emotion_name}' is disabled on this chassis."}

        # Check if emotion exists
        try:
            emotion_names = _safe_emotion_names()
            if not emotion_names:
                return {"error": "No emotions currently available"}

            if not emotion_name:
                emotion_name = random.choice(emotion_names)

            if emotion_name not in emotion_names:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            speed = getattr(deps.movement_manager, "speed_factor", 1.0)
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES, speed_factor=speed)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
