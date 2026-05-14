"""Tests for the ElevenLabs per-profile config files (elevenlabs.txt).

Verifies that every comedian persona has a committed elevenlabs.txt with the
required keys and valid values, and that no voice_id is committed (those go
in the gitignored elevenlabs.local.txt per persona).
"""

from __future__ import annotations
from pathlib import Path

import pytest


# The 8 comedian personas that must each have an elevenlabs.txt.
COMEDIAN_PERSONAS = [
    "andrew_dice_clay",
    "bill_hicks",
    "dave_chappelle",
    "don_rickles",
    "george_carlin",
    "richard_pryor",
    "robin_williams",
    "rodney_dangerfield",
]

REQUIRED_KEYS = {"voice", "stability", "similarity_boost"}

_PROFILES_ROOT = Path(__file__).parents[1] / "profiles"


def _parse_elevenlabs_txt(path: Path) -> dict[str, str]:
    """Parse a key=value config file, skipping comment and blank lines."""
    params: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            params[k.strip()] = v.strip()
    return params


@pytest.mark.parametrize("persona", COMEDIAN_PERSONAS)
def test_elevenlabs_txt_exists(persona: str) -> None:
    """Each comedian persona must have an elevenlabs.txt in its profile directory."""
    config_file = _PROFILES_ROOT / persona / "elevenlabs.txt"
    assert config_file.is_file(), f"Missing elevenlabs.txt for persona '{persona}' at {config_file}"


@pytest.mark.parametrize("persona", COMEDIAN_PERSONAS)
def test_elevenlabs_txt_parses_as_key_value(persona: str) -> None:
    """The elevenlabs.txt must be parseable as key=value lines."""
    config_file = _PROFILES_ROOT / persona / "elevenlabs.txt"
    params = _parse_elevenlabs_txt(config_file)
    # Must parse to at least one key (file is not empty / all comments).
    assert params, f"elevenlabs.txt for '{persona}' parsed to an empty dict — no key=value lines found"


@pytest.mark.parametrize("persona", COMEDIAN_PERSONAS)
def test_elevenlabs_txt_has_required_keys(persona: str) -> None:
    """Required keys (voice, stability, similarity_boost) must be present."""
    config_file = _PROFILES_ROOT / persona / "elevenlabs.txt"
    params = _parse_elevenlabs_txt(config_file)
    missing = REQUIRED_KEYS - params.keys()
    assert not missing, f"elevenlabs.txt for '{persona}' is missing required key(s): {missing}"


@pytest.mark.parametrize("persona", COMEDIAN_PERSONAS)
def test_elevenlabs_txt_stability_in_range(persona: str) -> None:
    """The stability value must be a float in [0, 1]."""
    config_file = _PROFILES_ROOT / persona / "elevenlabs.txt"
    params = _parse_elevenlabs_txt(config_file)
    raw = params.get("stability", "")
    try:
        value = float(raw)
    except ValueError:
        pytest.fail(f"elevenlabs.txt for '{persona}': stability={raw!r} is not a valid float")
    assert 0.0 <= value <= 1.0, f"elevenlabs.txt for '{persona}': stability={value} is out of range [0, 1]"


@pytest.mark.parametrize("persona", COMEDIAN_PERSONAS)
def test_elevenlabs_txt_similarity_boost_in_range(persona: str) -> None:
    """The similarity_boost value must be a float in [0, 1]."""
    config_file = _PROFILES_ROOT / persona / "elevenlabs.txt"
    params = _parse_elevenlabs_txt(config_file)
    raw = params.get("similarity_boost", "")
    try:
        value = float(raw)
    except ValueError:
        pytest.fail(f"elevenlabs.txt for '{persona}': similarity_boost={raw!r} is not a valid float")
    assert 0.0 <= value <= 1.0, f"elevenlabs.txt for '{persona}': similarity_boost={value} is out of range [0, 1]"


@pytest.mark.parametrize("persona", COMEDIAN_PERSONAS)
def test_elevenlabs_txt_no_committed_voice_id(persona: str) -> None:
    """No voice_id= line should be committed — those belong in elevenlabs.local.txt."""
    config_file = _PROFILES_ROOT / persona / "elevenlabs.txt"
    params = _parse_elevenlabs_txt(config_file)
    assert "voice_id" not in params, (
        f"elevenlabs.txt for '{persona}' has a committed voice_id={params['voice_id']!r}. "
        "Move it to the gitignored elevenlabs.local.txt instead."
    )
