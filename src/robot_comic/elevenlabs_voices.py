"""ElevenLabs voice catalog management.

Fetches available voices from the ElevenLabs /v1/voices API at startup and
caches them for the process lifetime. Falls back to hardcoded voices if the
API is unreachable.
"""

import logging

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

# Cache: filled by fetch_elevenlabs_voices() at startup
_voice_cache: dict[str, str] | None = None


async def fetch_elevenlabs_voices() -> dict[str, str]:
    """Fetch the ElevenLabs voice catalog via /v1/voices.

    Returns a {name: voice_id} mapping for all available voices (prebuilt +
    custom/PVC clones). Falls back to hardcoded prebuilt voices if the API
    is unreachable.

    The result is cached for the process lifetime — called once at startup,
    not per-turn.
    """
    global _voice_cache

    if _voice_cache is not None:
        return _voice_cache

    api_key = config.ELEVENLABS_API_KEY
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not configured; using fallback voice catalog")
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

            # Extract voices: each entry has "name" and "voice_id" fields.
            # Prebuilt voices come from the API; custom/PVC clones are user-owned.
            voices: dict[str, str] = {}
            if "voices" in data:
                for voice in data["voices"]:
                    name = voice.get("name")
                    voice_id = voice.get("voice_id")
                    if name and voice_id:
                        voices[name] = voice_id

            if voices:
                logger.info("Loaded %d voices from ElevenLabs API", len(voices))
                _voice_cache = voices
                return _voice_cache
            else:
                logger.warning("ElevenLabs API returned empty voice list; using fallback")
                _voice_cache = dict(_FALLBACK_VOICES)
                return _voice_cache

    except Exception as exc:
        logger.warning("Failed to fetch ElevenLabs voices: %s; using fallback", exc)
        _voice_cache = dict(_FALLBACK_VOICES)
        return _voice_cache


def get_elevenlabs_voices() -> dict[str, str]:
    """Return the cached voice catalog (populated by fetch_elevenlabs_voices).

    If not yet fetched (e.g., in a sync context), returns fallback voices.
    """
    if _voice_cache is not None:
        return _voice_cache
    return dict(_FALLBACK_VOICES)
