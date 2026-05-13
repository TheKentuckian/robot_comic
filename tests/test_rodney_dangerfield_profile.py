"""Regression tests for the Rodney Dangerfield persona profile.

These tests are purely file-content checks — they do not import or run any
runtime code.  Their job is to act as a sentinel so that future refactors
cannot silently hollow out the profile.
"""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILES_ROOT = Path(__file__).parents[1] / "profiles"
_PROFILE_DIR = _PROFILES_ROOT / "rodney_dangerfield"


def _read(filename: str) -> str:
    return (_PROFILE_DIR / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


def test_profile_directory_exists() -> None:
    """The rodney_dangerfield profile directory must be present."""
    assert _PROFILE_DIR.is_dir(), f"Missing profile directory: {_PROFILE_DIR}"


@pytest.mark.parametrize(
    "filename",
    [
        "instructions.txt",
        "tools.txt",
        "gemini_live.txt",
        "gemini_tts.txt",
        "elevenlabs.txt",
        "voice.txt",
    ],
)
def test_profile_file_exists(filename: str) -> None:
    """Every required profile file must exist and be non-empty."""
    path = _PROFILE_DIR / filename
    assert path.is_file(), f"Missing profile file: {path}"
    assert path.stat().st_size > 0, f"Profile file is empty: {path}"


# ---------------------------------------------------------------------------
# instructions.txt content checks
# ---------------------------------------------------------------------------


def test_instructions_contains_trademark_phrase() -> None:
    """instructions.txt must include the 'I get no respect' trademark phrase."""
    text = _read("instructions.txt")
    assert "no respect" in text.lower(), "instructions.txt is missing Rodney's 'I get no respect' trademark"


def test_instructions_contains_opening_sequence() -> None:
    """instructions.txt must define an OPENING SEQUENCE section."""
    text = _read("instructions.txt")
    assert "OPENING SEQUENCE" in text, "instructions.txt is missing the OPENING SEQUENCE section"


def test_instructions_contains_crowd_work_pattern() -> None:
    """instructions.txt must define a CROWD-WORK PATTERN section."""
    text = _read("instructions.txt")
    assert "CROWD-WORK PATTERN" in text, "instructions.txt is missing the CROWD-WORK PATTERN section"


def test_instructions_contains_physical_beats() -> None:
    """instructions.txt must include PHYSICAL BEATS / emotion-code mappings."""
    text = _read("instructions.txt")
    assert "PHYSICAL BEATS" in text, "instructions.txt is missing the PHYSICAL BEATS section"


def test_instructions_contains_disgusted1_emotion() -> None:
    """instructions.txt must map the disgusted1 emotion code (inward self-deprecation).

    This is a regression guard for the annotation added in PR #179, where
    'disgusted1' was specifically chosen because disgust is directed inward
    (at himself and at the pair of you) rather than at the mark.
    """
    text = _read("instructions.txt")
    assert "disgusted1" in text, (
        "instructions.txt is missing the 'disgusted1' emotion-code mapping "
        "(self-directed disgust, annotated in PR #179)"
    )


def test_instructions_contains_guardrails() -> None:
    """instructions.txt must contain guardrails preventing punching at the audience."""
    text = _read("instructions.txt")
    assert "GUARDRAILS" in text, "instructions.txt is missing the GUARDRAILS section"


# ---------------------------------------------------------------------------
# gemini_live.txt — regression guard (PR #188 audit)
# ---------------------------------------------------------------------------


def test_gemini_live_txt_is_non_empty() -> None:
    """gemini_live.txt must exist and contain delivery guidance (PR #188 regression)."""
    text = _read("gemini_live.txt")
    # Must contain at least one substantive sentence about pacing / delivery.
    assert len(text.strip()) > 50, "gemini_live.txt appears to be a stub — expected delivery guidance"


def test_gemini_live_txt_references_pace_or_delivery() -> None:
    """gemini_live.txt must include guidance on pacing or delivery rhythm."""
    text = _read("gemini_live.txt").lower()
    assert any(word in text for word in ("pace", "slow", "rhythm", "delivery", "baseline")), (
        "gemini_live.txt does not mention pacing or delivery — "
        "expected natural-language direction for the Gemini Live backend"
    )


# ---------------------------------------------------------------------------
# tools.txt — standard tool set
# ---------------------------------------------------------------------------


def test_tools_txt_contains_required_tools() -> None:
    """tools.txt must list the standard persona tools."""
    text = _read("tools.txt")
    for tool in ("greet", "crowd_work", "play_emotion", "move_head"):
        assert tool in text, f"tools.txt is missing required tool: {tool!r}"


# ---------------------------------------------------------------------------
# elevenlabs.txt — public config present (no real voice_id committed)
# ---------------------------------------------------------------------------


def test_elevenlabs_txt_has_voice_key() -> None:
    """elevenlabs.txt must define a default voice= key (public placeholder)."""
    text = _read("elevenlabs.txt")
    assert "voice=" in text, "elevenlabs.txt is missing a voice= key"


def test_elevenlabs_txt_does_not_contain_voice_id_value() -> None:
    """elevenlabs.txt must NOT commit a real voice_id= value (goes in .local.txt)."""
    text = _read("elevenlabs.txt")
    # Acceptable: commented-out line or no line at all.
    # Unacceptable: an uncommented voice_id=<something> line.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("voice_id=") or stripped == "voice_id=", (
            "elevenlabs.txt must not commit a real voice_id value — use elevenlabs.local.txt (gitignored) instead"
        )
