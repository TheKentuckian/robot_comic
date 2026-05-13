"""Tests for the George Carlin persona profile completeness and content quality."""

from pathlib import Path

import pytest


PROFILE_DIR = Path(__file__).parents[1] / "profiles" / "george_carlin"

REQUIRED_FILES = [
    "instructions.txt",
    "gemini_live.txt",
    "gemini_tts.txt",
    "tools.txt",
    "voice.txt",
    "elevenlabs.txt",
    "chatterbox.txt",
]


def test_profile_directory_exists() -> None:
    """The george_carlin profile directory must exist."""
    assert PROFILE_DIR.is_dir(), f"Profile directory not found: {PROFILE_DIR}"


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_file_exists(filename: str) -> None:
    """Each required profile file must exist."""
    path = PROFILE_DIR / filename
    assert path.is_file(), f"Missing required profile file: {path}"


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_file_nonempty(filename: str) -> None:
    """Each required profile file must not be empty."""
    path = PROFILE_DIR / filename
    content = path.read_text(encoding="utf-8").strip()
    assert content, f"Profile file is empty: {path}"


class TestInstructionsContent:
    """Verify instructions.txt captures Carlin's core signatures."""

    @pytest.fixture(scope="class")
    def instructions(self) -> str:
        return (PROFILE_DIR / "instructions.txt").read_text(encoding="utf-8").lower()

    def test_mentions_language(self, instructions: str) -> None:
        """Carlin's persona is built around language deconstruction."""
        assert "language" in instructions

    def test_mentions_euphemism(self, instructions: str) -> None:
        """Euphemism-targeting is a Carlin signature."""
        assert "euphemism" in instructions

    def test_mentions_seven_words_or_deconstruction(self, instructions: str) -> None:
        """References either the 'seven words' bit or deconstruction technique."""
        has_seven = "seven" in instructions
        has_deconstruct = "deconstruct" in instructions
        assert has_seven or has_deconstruct, "instructions.txt should reference 'seven' (Seven Words) or 'deconstruct'"

    def test_opening_sequence_present(self, instructions: str) -> None:
        """An OPENING SEQUENCE section must be defined."""
        assert "opening sequence" in instructions

    def test_crowd_work_pattern_present(self, instructions: str) -> None:
        """A CROWD-WORK PATTERN section must be defined."""
        assert "crowd-work" in instructions or "crowd_work" in instructions

    def test_physical_beats_present(self, instructions: str) -> None:
        """Physical beat / emotion-code mappings must be present."""
        assert "physical beats" in instructions or "play_emotion" in instructions

    def test_guardrails_present(self, instructions: str) -> None:
        """GUARDRAILS section must be present."""
        assert "guardrail" in instructions

    def test_emotion_codes_include_sensible_values(self, instructions: str) -> None:
        """Emotion codes should reference recognisable emotion code names."""
        assert "curious1" in instructions or "reprimand1" in instructions or "laughing1" in instructions

    def test_gemini_tts_tags_present(self, instructions: str) -> None:
        """GEMINI TTS DELIVERY TAGS section should be present."""
        assert "gemini tts" in instructions


class TestGeminiLiveContent:
    """Verify gemini_live.txt delivery guide is substantive."""

    @pytest.fixture(scope="class")
    def gemini_live(self) -> str:
        return (PROFILE_DIR / "gemini_live.txt").read_text(encoding="utf-8").lower()

    def test_mentions_pacing(self, gemini_live: str) -> None:
        """Must include pacing guidance."""
        assert "pace" in gemini_live or "pacing" in gemini_live or "deliberate" in gemini_live

    def test_mentions_punchline_delivery(self, gemini_live: str) -> None:
        """Must describe punchline delivery style."""
        assert "punchline" in gemini_live

    def test_mentions_escalation(self, gemini_live: str) -> None:
        """Must describe the escalation arc."""
        assert "escalat" in gemini_live


class TestToolsContent:
    """Verify tools.txt lists the expected tools."""

    @pytest.fixture(scope="class")
    def tools(self) -> list[str]:
        text = (PROFILE_DIR / "tools.txt").read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def test_includes_crowd_work(self, tools: list[str]) -> None:
        """crowd_work is the core engine for Carlin's persona."""
        assert "crowd_work" in tools

    def test_includes_play_emotion(self, tools: list[str]) -> None:
        """play_emotion must be available for physical-beat mappings."""
        assert "play_emotion" in tools

    def test_includes_greet(self, tools: list[str]) -> None:
        """greet is needed for the opening sequence scan."""
        assert "greet" in tools

    def test_no_roast_tool(self, tools: list[str]) -> None:
        """Carlin's persona does not use the roast tool (deconstruction, not roasting)."""
        assert "roast" not in tools


class TestElevenLabsConfig:
    """Verify elevenlabs.txt has the required config keys."""

    @pytest.fixture(scope="class")
    def config_lines(self) -> list[str]:
        text = (PROFILE_DIR / "elevenlabs.txt").read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]

    def test_voice_key_present(self, config_lines: list[str]) -> None:
        """voice= key must be set."""
        assert any(line.startswith("voice=") for line in config_lines)

    def test_stability_key_present(self, config_lines: list[str]) -> None:
        """stability= key must be present."""
        assert any(line.startswith("stability=") for line in config_lines)

    def test_similarity_boost_key_present(self, config_lines: list[str]) -> None:
        """similarity_boost= key must be present."""
        assert any(line.startswith("similarity_boost=") for line in config_lines)


class TestChatterboxConfig:
    """Verify chatterbox.txt has the required config keys."""

    @pytest.fixture(scope="class")
    def config_lines(self) -> list[str]:
        text = (PROFILE_DIR / "chatterbox.txt").read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]

    def test_voice_key_present(self, config_lines: list[str]) -> None:
        """voice= key must be set."""
        assert any(line.startswith("voice=") for line in config_lines)

    def test_exaggeration_key_present(self, config_lines: list[str]) -> None:
        """exaggeration= key must be present."""
        assert any(line.startswith("exaggeration=") for line in config_lines)

    def test_cfg_weight_key_present(self, config_lines: list[str]) -> None:
        """cfg_weight= key must be present."""
        assert any(line.startswith("cfg_weight=") for line in config_lines)
