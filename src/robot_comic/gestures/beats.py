"""Abstract beat names for per-persona gesture mapping.

A "beat" is an abstract comedic or conversational moment that personas
can map to different canonical gestures.  This indirection lets each
persona express the same dramatic beat in a physically distinct way:
Rickles shrugs dismissal outward; Dangerfield collapses inward.

Beat name constants
-------------------
Import these names to avoid magic strings::

    from robot_comic.gestures.beats import BEAT_DISAPPROVAL
    registry.resolve_for_persona(BEAT_DISAPPROVAL, persona_beats)

Canonical gesture names (from gestures.py)
-------------------------------------------
shrug, nod_yes, nod_no, point_left, point_right, scan, lean_in
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Canonical beat name constants
# ---------------------------------------------------------------------------

BEAT_DISAPPROVAL = "disapproval"
"""A "no, that's wrong" reaction — rejecting a target or premise."""

BEAT_AGREEMENT = "agreement"
"""A "yes, exactly" reaction — validating the user."""

BEAT_REFLECTION = "reflection"
"""A "let me think" beat — brief silent consideration before a response."""

BEAT_CHARACTER_SWITCH = "character_switch"
"""Transitioning into voicing another character or doing an impression."""

BEAT_PUNCHLINE_SETUP = "punchline_setup"
"""The beat just BEFORE delivering a punchline — build-up or pause."""

BEAT_PUNCHLINE_DROP = "punchline_drop"
"""Physical resolution AFTER the punchline lands."""

BEAT_DISMISSAL = "dismissal"
"""A "you're worthless, next" beat — sending someone off."""

BEAT_ACKNOWLEDGEMENT = "acknowledgement"
"""A "I see you, fellow human" beat — greet-style recognition."""

BEAT_SURPRISE = "surprise"
"""A sudden unexpected beat — caught off guard."""

BEAT_DEFEAT = "defeat"
"""A "I give up, what can you do" beat — resigned collapse."""

BEAT_SWAGGER = "swagger"
"""Confident wind-up to a delivery — the persona in full command."""

BEAT_VULNERABILITY = "vulnerability"
"""Opening up, lower stance — an honest or tender moment."""

# ---------------------------------------------------------------------------
# Full set — useful for validation
# ---------------------------------------------------------------------------

ALL_BEATS: frozenset[str] = frozenset(
    {
        BEAT_DISAPPROVAL,
        BEAT_AGREEMENT,
        BEAT_REFLECTION,
        BEAT_CHARACTER_SWITCH,
        BEAT_PUNCHLINE_SETUP,
        BEAT_PUNCHLINE_DROP,
        BEAT_DISMISSAL,
        BEAT_ACKNOWLEDGEMENT,
        BEAT_SURPRISE,
        BEAT_DEFEAT,
        BEAT_SWAGGER,
        BEAT_VULNERABILITY,
    }
)
