"""ElevenLabs voice catalog management.

Fetches available voices from the ElevenLabs /v1/voices API at startup and
caches them for the process lifetime. Falls back to hardcoded voices if the
API is unreachable.

Resolution helpers tolerate the API's decorated prebuilt names (e.g.
``"Brian - Deep, Resonant and Comforting"``) by matching configured short
names with a word-boundary prefix when an exact match misses.
"""

from __future__ import annotations
import logging
from typing import TypedDict

import httpx

from robot_comic.config import config


logger = logging.getLogger(__name__)

# Hardcoded fallback: Prebuilt ElevenLabs voices from Turbo v2.5 standard set
# mapped to their canonical voice IDs. Used when the /v1/voices API is
# unreachable; mapping a name to itself would 400/404 against /text-to-speech/<id>.
_FALLBACK_VOICES: dict[str, str] = {
    "Rachel": "21m00Tcm4ijWNoXd58YU",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Antoni": "ErXwobaYiN019PkySvjV",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Elli": "MF3mGyEYCl7XYWbV9V6O",
    "Gigi": "jBpfuIE2acCO8z3wKNLl",
    "Freya": "jsCqWAovK2LkecY7zXl4",
    "Harry": "SOYHLrjzK2X1ezoPC6cr",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "River": "SAz9YHcvj6GT2YYXdXww",
    "Sam": "yoZ06aMxZJJ28mfd3POQ",
}

# Default fallback short name used when name resolution misses. Kept aligned
# with config.ELEVENLABS_DEFAULT_VOICE but referenced here so this module can
# be used without dragging the full config in. Resolution always tries prefix
# matching for the fallback too, so ``Adam`` will resolve to
# ``"Adam - Dominant, Firm"`` if that is what the API returns.
DEFAULT_FALLBACK_SHORT_NAME = "Adam"


class VoiceRecord(TypedDict):
    """A single voice entry as returned from /v1/voices."""

    name: str
    voice_id: str
    category: str


# Caches: filled by fetch_elevenlabs_voices() at startup.
# - _voice_cache preserves the legacy {name: voice_id} shape for callers that
#   only need the mapping (e.g. admin-UI voice picker).
# - _voice_records_cache preserves the richer per-voice records (name, id,
#   category) needed by resolve_voice_id_by_name's word-boundary + last-resort
#   premade fallback.
_voice_cache: dict[str, str] | None = None
_voice_records_cache: list[VoiceRecord] | None = None


def _fallback_records() -> list[VoiceRecord]:
    """Build VoiceRecord list from the hardcoded fallback mapping."""
    return [{"name": name, "voice_id": voice_id, "category": "premade"} for name, voice_id in _FALLBACK_VOICES.items()]


async def fetch_elevenlabs_voices() -> dict[str, str]:
    """Fetch the ElevenLabs voice catalog via /v1/voices.

    Returns a {name: voice_id} mapping for all available voices (prebuilt +
    custom/PVC clones). Falls back to hardcoded prebuilt voices if the API
    is unreachable.

    The result is cached for the process lifetime — called once at startup,
    not per-turn. Side effect: also populates the richer
    ``_voice_records_cache`` used by :func:`resolve_voice_id_by_name`.
    """
    global _voice_cache, _voice_records_cache

    if _voice_cache is not None:
        return _voice_cache

    api_key = config.ELEVENLABS_API_KEY
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not configured; using fallback voice catalog")
        _voice_records_cache = _fallback_records()
        _voice_cache = dict(_FALLBACK_VOICES)
        return _voice_cache

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key},
            )
            response.raise_for_status()
            data = response.json()

            # Extract voices: each entry has "name", "voice_id", and "category" fields.
            # Prebuilt voices come from the API; custom/PVC clones are user-owned.
            voices: dict[str, str] = {}
            records: list[VoiceRecord] = []
            if "voices" in data:
                for voice in data["voices"]:
                    name = voice.get("name")
                    voice_id = voice.get("voice_id")
                    category = voice.get("category") or ""
                    if name and voice_id:
                        voices[name] = voice_id
                        records.append({"name": name, "voice_id": voice_id, "category": category})

            if voices:
                logger.info("Loaded %d voices from ElevenLabs API", len(voices))
                _voice_records_cache = records
                _voice_cache = voices
                return _voice_cache
            else:
                logger.warning("ElevenLabs API returned empty voice list; using fallback")
                _voice_records_cache = _fallback_records()
                _voice_cache = dict(_FALLBACK_VOICES)
                return _voice_cache

    except Exception as exc:
        logger.warning("Failed to fetch ElevenLabs voices: %s; using fallback", exc)
        _voice_records_cache = _fallback_records()
        _voice_cache = dict(_FALLBACK_VOICES)
        return _voice_cache


def get_elevenlabs_voices() -> dict[str, str]:
    """Return the cached voice catalog (populated by fetch_elevenlabs_voices).

    If not yet fetched (e.g., in a sync context), returns fallback voices.
    """
    if _voice_cache is not None:
        return _voice_cache
    return dict(_FALLBACK_VOICES)


def get_elevenlabs_voice_records() -> list[VoiceRecord]:
    """Return the cached voice records (name + voice_id + category).

    If not yet fetched (e.g., in a sync context), returns fallback records
    derived from the hardcoded prebuilt list (all marked ``category="premade"``).
    """
    if _voice_records_cache is not None:
        return list(_voice_records_cache)
    return _fallback_records()


def _name_matches_prefix(short_name: str, candidate: str) -> bool:
    """Return True when ``candidate`` starts with ``short_name`` at a word boundary.

    ``"Brian"`` matches ``"Brian - Deep, Resonant…"`` and ``"Brian-X"`` but not
    ``"Brianna - Sweet"``. The next character after the short name must be one
    of: whitespace, hyphen, underscore, or end-of-string. Bare names with no
    decoration are handled by the exact-match path; this helper is for the
    decorated-name case.
    """
    if not short_name or not candidate:
        return False
    if not candidate.startswith(short_name):
        return False
    if len(candidate) == len(short_name):
        # Exact match — covered by the caller's exact-match step, but harmless
        # to return True here too.
        return True
    next_char = candidate[len(short_name)]
    return next_char.isspace() or next_char in ("-", "_")


def _lookup(
    short_name: str,
    records: list[VoiceRecord],
    case_insensitive: bool,
) -> str | None:
    """Try exact match then word-boundary prefix match for ``short_name``.

    When ``case_insensitive`` is True, both the target and the candidate names
    are lowercased before comparison.
    """
    target = short_name.lower() if case_insensitive else short_name
    # Pass 1: exact match
    for rec in records:
        cand = rec["name"].lower() if case_insensitive else rec["name"]
        if cand == target:
            return rec["voice_id"]
    # Pass 2: word-boundary prefix
    for rec in records:
        cand = rec["name"].lower() if case_insensitive else rec["name"]
        if _name_matches_prefix(target, cand):
            return rec["voice_id"]
    return None


def resolve_voice_id_by_name(
    short_name: str,
    fallback_name: str = DEFAULT_FALLBACK_SHORT_NAME,
    records: list[VoiceRecord] | None = None,
) -> str | None:
    """Resolve a short voice name to an ElevenLabs voice ID.

    Resolution order:

    1. Exact match against ``name``.
    2. Word-boundary prefix match (``"Brian"`` matches ``"Brian - Deep…"`` but
       not ``"Brianna - Sweet"``).
    3. Case-insensitive variants of (1) and (2).
    4. The same logic applied to ``fallback_name`` (default ``"Adam"``).
    5. The first ``category="premade"`` voice in the catalog as an absolute
       last resort. A warning is logged with the configured short name and a
       truncated catalog preview so the operator can diagnose the miss.

    Returns ``None`` only when the catalog is empty (no voices loaded at all).
    """
    cat = list(records) if records is not None else get_elevenlabs_voice_records()

    if not cat:
        return None

    # Steps 1+2 (case-sensitive), then steps 1+2 (case-insensitive)
    if short_name:
        hit = _lookup(short_name, cat, case_insensitive=False)
        if hit is not None:
            return hit
        hit = _lookup(short_name, cat, case_insensitive=True)
        if hit is not None:
            return hit

    # Step 4: same logic on the fallback
    if fallback_name and fallback_name != short_name:
        hit = _lookup(fallback_name, cat, case_insensitive=False)
        if hit is not None:
            return hit
        hit = _lookup(fallback_name, cat, case_insensitive=True)
        if hit is not None:
            return hit

    # Step 5: last resort — first premade voice. Surface enough context so the
    # operator can see what was available and what was being asked for.
    preview = ", ".join(rec["name"] for rec in cat[:10])
    suffix = "" if len(cat) <= 10 else f", … ({len(cat) - 10} more)"
    premade = next((rec for rec in cat if rec.get("category") == "premade"), None)
    if premade is not None:
        logger.warning(
            "No ElevenLabs voice matched %r (fallback %r also missed); using first premade voice %r. Available: %s%s",
            short_name,
            fallback_name,
            premade["name"],
            preview,
            suffix,
        )
        return premade["voice_id"]

    # No premade voices in the catalog — pick the first one we have rather
    # than returning None and surfacing the original opaque error.
    logger.warning(
        "No ElevenLabs voice matched %r (fallback %r also missed) and no premade "
        "voices in catalog; using first available voice %r. Available: %s%s",
        short_name,
        fallback_name,
        cat[0]["name"],
        preview,
        suffix,
    )
    return cat[0]["voice_id"]
