"""Tests for the ElevenLabs voice name resolver.

The resolver bridges short profile names (``voice=Brian``) and the decorated
names ElevenLabs returns from /v1/voices (``"Brian - Deep, Resonant and
Comforting"``). Resolution order is documented on
``resolve_voice_id_by_name``; these tests pin each step.
"""

from __future__ import annotations
import logging

import pytest

from robot_comic.elevenlabs_voices import (
    VoiceRecord,
    resolve_voice_id_by_name,
)


def _records(*pairs: tuple[str, str, str]) -> list[VoiceRecord]:
    """Build a VoiceRecord list from (name, voice_id, category) tuples."""
    return [{"name": name, "voice_id": vid, "category": cat} for name, vid, cat in pairs]


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


def test_resolver_exact_match_wins_over_prefix() -> None:
    """Exact name match is preferred over a prefix candidate.

    A catalog containing both ``"Brian"`` (exact) and ``"Brian - Deep…"``
    (decorated) must resolve ``Brian`` to the exact entry, not the decorated
    one. Order in the catalog is intentionally decorated-first to ensure the
    resolver isn't just returning the first match it sees.
    """
    cat = _records(
        ("Brian - Deep, Resonant and Comforting", "id_decorated", "premade"),
        ("Brian", "id_exact", "premade"),
    )
    assert resolve_voice_id_by_name("Brian", records=cat) == "id_exact"


# ---------------------------------------------------------------------------
# Prefix match with word boundary
# ---------------------------------------------------------------------------


def test_resolver_prefix_match_resolves_decorated_name() -> None:
    """``voice=Brian`` matches ``"Brian - Deep, Resonant and Comforting"``."""
    cat = _records(
        ("Brian - Deep, Resonant and Comforting", "nPczCjzI2devNBz1zQrb", "premade"),
        ("Adam - Dominant, Firm", "pNInz6obpgDQGcFmaJgB", "premade"),
    )
    assert resolve_voice_id_by_name("Brian", records=cat) == "nPczCjzI2devNBz1zQrb"


def test_resolver_prefix_match_word_boundary_discriminates_brianna() -> None:
    """``Brian`` must NOT match ``"Brianna - Sweet"`` — the word-boundary check
    catches the substring trap that a naive ``startswith`` would walk into.
    Falls through to the Adam fallback instead.
    """
    cat = _records(
        ("Brianna - Sweet", "id_brianna", "premade"),
        ("Adam - Dominant, Firm", "id_adam", "premade"),
    )
    # Falls through to "Adam" fallback (prefix-matched), not Brianna.
    assert resolve_voice_id_by_name("Brian", records=cat) == "id_adam"


def test_resolver_prefix_match_accepts_hyphen_and_underscore_boundary() -> None:
    """Both ``Brian-X`` and ``Brian_X`` are valid boundary forms.

    The API uses ``" - "`` as the standard separator but we accept hyphen and
    underscore directly too so custom-named voices don't have to match the
    exact decoration scheme.
    """
    for decorated in ("Brian-X", "Brian_X", "Brian Mk2"):
        cat = _records((decorated, "id_x", "premade"))
        assert resolve_voice_id_by_name("Brian", records=cat) == "id_x", (
            f"prefix match should accept boundary in {decorated!r}"
        )


# ---------------------------------------------------------------------------
# Case-insensitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["brian", "BRIAN", "BrIaN"])
def test_resolver_case_insensitive_prefix_match(query: str) -> None:
    """Profile values with non-canonical casing still resolve via case-insensitive prefix."""
    cat = _records(
        ("Brian - Deep, Resonant and Comforting", "id_brian", "premade"),
    )
    assert resolve_voice_id_by_name(query, records=cat) == "id_brian"


def test_resolver_case_sensitive_takes_priority_over_case_insensitive() -> None:
    """If a case-sensitive match exists, it beats a case-insensitive one.

    Catalog has ``"brian"`` (lowercase exact match for ``"brian"``) and
    ``"Brian - …"`` (decorated). Query ``"brian"`` should pick the exact
    lowercase entry, not the case-insensitive prefix on ``"Brian - …"``.
    """
    cat = _records(
        ("Brian - Deep, Resonant and Comforting", "id_decorated", "premade"),
        ("brian", "id_lower_exact", "premade"),
    )
    assert resolve_voice_id_by_name("brian", records=cat) == "id_lower_exact"


# ---------------------------------------------------------------------------
# Fallback applies the same prefix logic
# ---------------------------------------------------------------------------


def test_resolver_fallback_uses_prefix_match_too() -> None:
    """The fallback name ``Adam`` resolves to ``"Adam - Dominant, Firm"``.

    Pre-fix, the fallback only did exact match and so failed for the same
    reason the primary lookup did — this is the production bug.
    """
    cat = _records(
        ("Adam - Dominant, Firm", "pNInz6obpgDQGcFmaJgB", "premade"),
        ("Bill - Wise, Mature, Balanced", "pqHfZKP75CvOlQylNhV4", "premade"),
    )
    # "Charlie" is in nobody's catalog — falls through to Adam-with-prefix.
    assert resolve_voice_id_by_name("Charlie", fallback_name="Adam", records=cat) == "pNInz6obpgDQGcFmaJgB"


def test_resolver_fallback_skipped_when_same_as_short_name() -> None:
    """If the short name and fallback are identical, we don't re-run the same
    lookup — the resolver moves on to the last-resort premade step."""
    cat = _records(
        ("Bill - Wise, Mature, Balanced", "id_bill", "premade"),
    )
    # short_name=Charlie, fallback=Charlie → both miss → premade fallback
    assert resolve_voice_id_by_name("Charlie", fallback_name="Charlie", records=cat) == "id_bill"


# ---------------------------------------------------------------------------
# All-miss last-resort path
# ---------------------------------------------------------------------------


def test_resolver_all_miss_returns_first_premade_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """When nothing matches and the fallback also misses, the resolver returns
    the first ``category="premade"`` voice and logs a diagnostic line naming
    the requested short name and a truncated catalog preview.
    """
    cat = _records(
        ("custom_clone_1", "id_clone1", "cloned"),
        ("Bill - Wise, Mature, Balanced", "id_bill", "premade"),
        ("Charlie - Casual", "id_charlie", "premade"),
    )
    with caplog.at_level(logging.WARNING, logger="robot_comic.elevenlabs_voices"):
        result = resolve_voice_id_by_name("DoesNotExist", fallback_name="AlsoMissing", records=cat)
    # First premade in the list wins.
    assert result == "id_bill"
    # Log mentions both the requested name and at least one available voice.
    assert any("DoesNotExist" in rec.message for rec in caplog.records)
    assert any("Bill" in rec.message for rec in caplog.records)


def test_resolver_all_miss_no_premade_returns_first_voice() -> None:
    """If the catalog is non-empty but contains zero premade voices (e.g. an
    account with only PVC clones), the resolver still returns *something*
    rather than None — the first voice in the catalog wins."""
    cat = _records(
        ("pvc_clone_a", "id_a", "cloned"),
        ("pvc_clone_b", "id_b", "cloned"),
    )
    assert resolve_voice_id_by_name("DoesNotExist", fallback_name="AlsoMissing", records=cat) == "id_a"


def test_resolver_empty_catalog_returns_none() -> None:
    """An empty catalog (no voices loaded at all) returns None — the caller
    needs to surface a clear error rather than fabricate an ID."""
    assert resolve_voice_id_by_name("Brian", records=[]) is None


# ---------------------------------------------------------------------------
# Word-boundary edge cases
# ---------------------------------------------------------------------------


def test_resolver_prefix_does_not_match_naive_substring_inside_word() -> None:
    """``Bri`` does NOT match ``"Brian - …"`` — the boundary check requires the
    short name end at a real separator, not in the middle of a word."""
    cat = _records(
        ("Brian - Deep, Resonant and Comforting", "id_brian", "premade"),
        ("Adam - Dominant, Firm", "id_adam", "premade"),
    )
    # Falls through to Adam fallback.
    assert resolve_voice_id_by_name("Bri", records=cat) == "id_adam"
