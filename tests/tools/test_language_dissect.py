"""Tests for the LanguageDissect tool and the euphemisms dictionary."""

from __future__ import annotations
import sys
import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_TOOL_PATH = (
    Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "language_dissect.py"
)
_EUPHEMISMS_PATH = (
    Path(__file__).parents[2] / "profiles" / "george_carlin" / "euphemisms.json"
)


def _load_language_dissect_module():
    """Load language_dissect from its source path, bypassing the package registry."""
    spec = importlib.util.spec_from_file_location("language_dissect_test_module", _TOOL_PATH)
    assert spec and spec.loader, f"Cannot load module from {_TOOL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["language_dissect_test_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def ld_module():
    return _load_language_dissect_module()


@pytest.fixture(scope="module")
def euphemisms_dict():
    """Load the raw euphemisms JSON for independent validation."""
    with _EUPHEMISMS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_deps() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Dictionary integrity
# ---------------------------------------------------------------------------


class TestEuphemismsDictionary:
    def test_dict_loads_as_mapping(self, euphemisms_dict):
        """euphemisms.json must parse as a non-empty dict."""
        assert isinstance(euphemisms_dict, dict)
        assert len(euphemisms_dict) >= 30, (
            f"Expected at least 30 entries, got {len(euphemisms_dict)}"
        )

    def test_every_entry_has_required_keys(self, euphemisms_dict):
        """Every entry must have literal_words, euphemism_target, dissection_suggestion."""
        required = {"literal_words", "euphemism_target", "dissection_suggestion"}
        for phrase, entry in euphemisms_dict.items():
            missing = required - entry.keys()
            assert not missing, (
                f"Entry '{phrase}' is missing keys: {missing}"
            )

    def test_literal_words_is_dict(self, euphemisms_dict):
        """literal_words must be a dict mapping word strings to meaning strings."""
        for phrase, entry in euphemisms_dict.items():
            lw = entry["literal_words"]
            assert isinstance(lw, dict), (
                f"Entry '{phrase}': literal_words must be a dict, got {type(lw).__name__}"
            )
            for word, meaning in lw.items():
                assert isinstance(word, str) and word, (
                    f"Entry '{phrase}': literal_words key must be non-empty string"
                )
                assert isinstance(meaning, str) and meaning, (
                    f"Entry '{phrase}': literal_words value must be non-empty string"
                )

    def test_euphemism_target_is_non_empty_string(self, euphemisms_dict):
        for phrase, entry in euphemisms_dict.items():
            target = entry["euphemism_target"]
            assert isinstance(target, str) and target.strip(), (
                f"Entry '{phrase}': euphemism_target must be a non-empty string"
            )

    def test_dissection_suggestion_is_non_empty_string(self, euphemisms_dict):
        for phrase, entry in euphemisms_dict.items():
            suggestion = entry["dissection_suggestion"]
            assert isinstance(suggestion, str) and suggestion.strip(), (
                f"Entry '{phrase}': dissection_suggestion must be a non-empty string"
            )

    def test_known_phrases_present(self, euphemisms_dict):
        """A selection of canonical Carlin targets must be in the dictionary."""
        expected = [
            "thoughts and prayers",
            "passed away",
            "human resources",
            "collateral damage",
            "enhanced interrogation",
            "downsizing",
            "correctional facility",
        ]
        for phrase in expected:
            assert phrase in euphemisms_dict, (
                f"Expected canonical phrase '{phrase}' not found in euphemisms.json"
            )


# ---------------------------------------------------------------------------
# dissect_phrase — known entry
# ---------------------------------------------------------------------------


class TestDissectPhraseKnown:
    def test_returns_expected_structure(self, ld_module):
        """dissect_phrase returns all four required keys for a known phrase."""
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert set(result.keys()) >= {"phrase", "literal_words", "euphemism_target", "dissection_suggestion"}

    def test_phrase_field_preserved(self, ld_module):
        """The phrase field echoes the original input, not the normalised version."""
        original = "Thoughts and Prayers"
        result = ld_module.dissect_phrase(original)
        assert result["phrase"] == original

    def test_literal_words_is_dict(self, ld_module):
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert isinstance(result["literal_words"], dict)
        assert len(result["literal_words"]) >= 1

    def test_euphemism_target_populated(self, ld_module):
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert isinstance(result["euphemism_target"], str)
        assert result["euphemism_target"] != "unknown"

    def test_dissection_suggestion_populated(self, ld_module):
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert isinstance(result["dissection_suggestion"], str)
        assert result["dissection_suggestion"].strip()

    def test_case_insensitive_lookup(self, ld_module):
        """Lookup must be case-insensitive."""
        lower = ld_module.dissect_phrase("passed away")
        upper = ld_module.dissect_phrase("PASSED AWAY")
        mixed = ld_module.dissect_phrase("Passed Away")
        assert lower["euphemism_target"] == upper["euphemism_target"] == mixed["euphemism_target"]

    def test_human_resources_target(self, ld_module):
        """human resources — the euphemism_target should describe interchangeable workers."""
        result = ld_module.dissect_phrase("human resources")
        assert result["euphemism_target"] != "unknown"
        assert "literal_words" in result
        assert "human" in result["literal_words"] or "resources" in result["literal_words"]

    def test_collateral_damage_literal_words(self, ld_module):
        """collateral damage should have at least 'collateral' and 'damage' in literal_words."""
        result = ld_module.dissect_phrase("collateral damage")
        assert "collateral" in result["literal_words"]
        assert "damage" in result["literal_words"]


# ---------------------------------------------------------------------------
# dissect_phrase — unknown phrase fallback
# ---------------------------------------------------------------------------


class TestDissectPhraseUnknown:
    def test_returns_all_required_keys(self, ld_module):
        """Unknown phrase still returns all four structural keys."""
        result = ld_module.dissect_phrase("synergistic paradigm shift")
        assert set(result.keys()) >= {"phrase", "literal_words", "euphemism_target", "dissection_suggestion"}

    def test_phrase_field_preserved(self, ld_module):
        phrase = "synergistic paradigm shift"
        result = ld_module.dissect_phrase(phrase)
        assert result["phrase"] == phrase

    def test_literal_words_contains_each_word(self, ld_module):
        """For an unknown phrase, literal_words should have one entry per word."""
        result = ld_module.dissect_phrase("synergistic paradigm shift")
        lw = result["literal_words"]
        assert isinstance(lw, dict)
        assert "synergistic" in lw
        assert "paradigm" in lw
        assert "shift" in lw

    def test_euphemism_target_indicates_unknown(self, ld_module):
        """Unknown phrases should signal that the phrase is not in the dictionary."""
        result = ld_module.dissect_phrase("totally made up phrase")
        assert "unknown" in result["euphemism_target"].lower()

    def test_dissection_suggestion_non_empty(self, ld_module):
        result = ld_module.dissect_phrase("totally made up phrase")
        assert isinstance(result["dissection_suggestion"], str)
        assert result["dissection_suggestion"].strip()

    def test_single_word_phrase(self, ld_module):
        """A single unknown word should produce a single-key literal_words dict."""
        result = ld_module.dissect_phrase("quibble")
        lw = result["literal_words"]
        assert "quibble" in lw

    def test_empty_phrase_returns_error(self, ld_module):
        """Empty string should not crash — the tool layer handles it, but dissect_phrase
        should still return a dict (the tool wraps it in an error check)."""
        # dissect_phrase with empty string produces an empty literal_words dict
        result = ld_module.dissect_phrase("   ")
        # Whitespace-only normalises to empty — just verify it doesn't raise
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# LanguageDissect tool class
# ---------------------------------------------------------------------------


class TestLanguageDissectTool:
    @pytest.fixture
    def tool(self, ld_module):
        return ld_module.LanguageDissect()

    def test_name(self, tool):
        assert tool.name == "language_dissect"

    def test_description_non_empty(self, tool):
        assert isinstance(tool.description, str) and tool.description.strip()

    def test_parameters_schema_has_phrase(self, tool):
        props = tool.parameters_schema.get("properties", {})
        assert "phrase" in props
        assert tool.parameters_schema.get("required") == ["phrase"]

    @pytest.mark.asyncio
    async def test_call_known_phrase(self, tool):
        """Tool returns expected structure for a known euphemism via async __call__."""
        result = await tool(make_deps(), phrase="thoughts and prayers")
        assert result["phrase"] == "thoughts and prayers"
        assert isinstance(result["literal_words"], dict)
        assert result["euphemism_target"] != "unknown"
        assert result["dissection_suggestion"].strip()

    @pytest.mark.asyncio
    async def test_call_unknown_phrase(self, tool):
        """Tool returns reasonable defaults for an unknown phrase."""
        result = await tool(make_deps(), phrase="leveraged synergy")
        assert result["phrase"] == "leveraged synergy"
        assert isinstance(result["literal_words"], dict)
        assert "leveraged" in result["literal_words"]
        assert "synergy" in result["literal_words"]
        assert "unknown" in result["euphemism_target"].lower()

    @pytest.mark.asyncio
    async def test_call_empty_phrase_returns_error(self, tool):
        """Empty phrase argument returns an error dict without raising."""
        result = await tool(make_deps(), phrase="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_call_missing_phrase_returns_error(self, tool):
        """Missing phrase kwarg returns an error dict without raising."""
        result = await tool(make_deps())
        assert "error" in result

    def test_spec_structure(self, tool):
        """spec() returns a well-formed function-call spec."""
        spec = tool.spec()
        assert spec["type"] == "function"
        assert spec["name"] == "language_dissect"
        assert "description" in spec
        assert "parameters" in spec
