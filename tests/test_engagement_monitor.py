"""Unit tests for EngagementMonitor (guardrail.py).

Pure unit tests — no LLM, no audio, no network.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic.guardrail import (
    SOFTEN_NOTE,
    SOFTEN_NOTES,
    GUARDRAIL_PROFILES,
    EngagementMonitor,
    get_soften_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monitor(profile: str = "bill_hicks", **env_overrides: str) -> EngagementMonitor:
    """Return a fresh EngagementMonitor; ``env_overrides`` patch os.environ."""
    return EngagementMonitor(profile=profile)


# ---------------------------------------------------------------------------
# Scoring: single-turn discomfort signal
# ---------------------------------------------------------------------------


class TestDiscomfortScoring:
    """Verify that individual discomfort phrases produce score > 0.5."""

    @pytest.mark.parametrize(
        "text",
        [
            "stop",
            "okay enough",
            "that's enough",
            "can we change the topic",
            "tone it down",
            "less aggressive please",
            "not funny",
            "I'm uncomfortable",
            "please stop",
            "let's change",
            "just stop already",
            "go away",
        ],
    )
    def test_discomfort_phrase_scores_above_half(self, text: str) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        score, _ = monitor.analyze(text)
        assert score > 0.5, f"Expected score > 0.5 for {text!r}, got {score}"

    def test_empty_input_scores_above_half(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        score, _ = monitor.analyze("")
        assert score > 0.5

    def test_blank_whitespace_scores_above_half(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        score, _ = monitor.analyze("   ")
        assert score > 0.5

    def test_positive_engagement_scores_zero(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        score, _ = monitor.analyze("That's a fascinating point. I never thought about it that way before.")
        assert score == 0.0

    def test_short_response_scores_below_half(self) -> None:
        # A very short response is a weak signal (<0.5) — not enough alone to soften.
        monitor = EngagementMonitor(profile="bill_hicks")
        score, _ = monitor.analyze("Okay.")
        assert 0.0 < score < 0.5


# ---------------------------------------------------------------------------
# Consecutive-discomfort logic
# ---------------------------------------------------------------------------


class TestConsecutiveDiscomfort:
    """Verify that should_soften fires after N consecutive discomfort turns."""

    def test_single_discomfort_does_not_soften(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        _, should_soften = monitor.analyze("stop")
        assert should_soften is False

    def test_two_consecutive_discomfort_softens(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        monitor.analyze("stop")
        _, should_soften = monitor.analyze("enough already")
        assert should_soften is True

    def test_counter_resets_on_positive_turn(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        monitor.analyze("stop")
        assert monitor.consecutive_discomfort == 1

        # Positive turn resets counter.
        monitor.analyze("Actually that's a really interesting question.")
        assert monitor.consecutive_discomfort == 0

        # Need two more consecutive to re-trigger.
        monitor.analyze("stop")
        _, should_soften = monitor.analyze("I give up")
        assert should_soften is True

    def test_three_consecutive_discomfort_still_softens(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        monitor.analyze("stop")
        monitor.analyze("enough")
        _, should_soften = monitor.analyze("go away")
        assert should_soften is True

    def test_counter_increments_monotonically_while_discomfort(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        for i in range(1, 5):
            monitor.analyze("stop")
            assert monitor.consecutive_discomfort == i


# ---------------------------------------------------------------------------
# Profile-awareness / enable flag
# ---------------------------------------------------------------------------


class TestProfileAwareness:
    """Verify that the guardrail is only active for opted-in profiles."""

    def test_bill_hicks_enabled_by_default(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        assert monitor.enabled is True
        assert "bill_hicks" in GUARDRAIL_PROFILES

    def test_andrew_dice_clay_enabled_by_default(self) -> None:
        monitor = EngagementMonitor(profile="andrew_dice_clay")
        assert monitor.enabled is True
        assert "andrew_dice_clay" in GUARDRAIL_PROFILES

    def test_richard_pryor_enabled_by_default(self) -> None:
        monitor = EngagementMonitor(profile="richard_pryor")
        assert monitor.enabled is True
        assert "richard_pryor" in GUARDRAIL_PROFILES

    @pytest.mark.parametrize("profile", ["bill_hicks", "andrew_dice_clay", "richard_pryor"])
    def test_guardrail_activates_for_all_three_personas(self, profile: str) -> None:
        monitor = EngagementMonitor(profile=profile)
        assert monitor.enabled is True
        monitor.analyze("stop")
        _, should_soften = monitor.analyze("enough")
        assert should_soften is True

    @pytest.mark.parametrize("profile", ["bill_hicks", "andrew_dice_clay", "richard_pryor"])
    def test_persona_specific_soften_note_used(self, profile: str) -> None:
        note = get_soften_note(profile)
        assert note == SOFTEN_NOTES[profile], (
            f"Expected persona-specific note for {profile!r}"
        )

    def test_other_profile_disabled_by_default(self) -> None:
        monitor = EngagementMonitor(profile="don_rickles")
        assert monitor.enabled is False

    def test_profile_not_in_list_does_not_activate(self) -> None:
        """A persona not in GUARDRAIL_PROFILES must be a no-op."""
        monitor = EngagementMonitor(profile="george_carlin")
        assert monitor.enabled is False
        score, should_soften = monitor.analyze("stop everything right now")
        assert score == 0.0
        assert should_soften is False

    def test_disabled_monitor_always_returns_zero_false(self) -> None:
        monitor = EngagementMonitor(profile="don_rickles")
        score, should_soften = monitor.analyze("stop everything right now")
        assert score == 0.0
        assert should_soften is False

    def test_env_override_force_disable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REACHY_MINI_GUARDRAIL_ENABLED", "0")
        monitor = EngagementMonitor(profile="bill_hicks")
        assert monitor.enabled is False
        score, should_soften = monitor.analyze("stop")
        assert score == 0.0
        assert should_soften is False

    def test_env_override_force_enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REACHY_MINI_GUARDRAIL_ENABLED", "1")
        monitor = EngagementMonitor(profile="don_rickles")
        assert monitor.enabled is True

    def test_update_profile_resets_state(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        monitor.analyze("stop")
        assert monitor.consecutive_discomfort == 1

        monitor.update_profile("don_rickles")
        assert monitor.enabled is False
        assert monitor.consecutive_discomfort == 0


# ---------------------------------------------------------------------------
# Soften notes
# ---------------------------------------------------------------------------


class TestSoftenNotes:
    """Verify persona-specific and generic soften notes."""

    def test_bill_hicks_soften_note(self) -> None:
        note = get_soften_note("bill_hicks")
        assert "observational" in note.lower()
        assert "uncomfortable" in note.lower()

    def test_andrew_dice_clay_soften_note(self) -> None:
        note = get_soften_note("andrew_dice_clay")
        assert "misogyny" in note.lower()
        assert "uncomfortable" in note.lower()

    def test_richard_pryor_soften_note(self) -> None:
        note = get_soften_note("richard_pryor")
        assert "vulnerability" in note.lower()
        assert "uncomfortable" in note.lower()

    def test_unknown_persona_returns_generic(self) -> None:
        note = get_soften_note("some_random_persona")
        assert "ease back" in note.lower() or "lighter" in note.lower()

    def test_none_persona_returns_generic(self) -> None:
        note = get_soften_note(None)
        assert len(note) > 20

    def test_soften_note_backwards_compat(self) -> None:
        """SOFTEN_NOTE alias still points to the bill_hicks note."""
        assert SOFTEN_NOTE == SOFTEN_NOTES["bill_hicks"]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_consecutive_counter(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        monitor.analyze("stop")
        monitor.analyze("enough")
        assert monitor.consecutive_discomfort == 2
        monitor.reset()
        assert monitor.consecutive_discomfort == 0
        assert monitor.last_score == 0.0

    def test_reset_does_not_change_enabled(self) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        assert monitor.enabled is True
        monitor.reset()
        assert monitor.enabled is True


# ---------------------------------------------------------------------------
# LLM scoring
# ---------------------------------------------------------------------------


class TestLLMScoring:
    """Verify score_via_llm and analyze() with LLM scoring enabled."""

    @pytest.mark.asyncio
    async def test_score_via_llm_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_via_llm returns the parsed float from the LLM response."""
        monitor = EngagementMonitor(profile="bill_hicks")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": '{"score": 0.85, "reason": "user asked to stop"}'}
        mock_resp.raise_for_status = MagicMock()

        http_client = AsyncMock()
        http_client.post = AsyncMock(return_value=mock_resp)

        score = await monitor.score_via_llm("please stop this", http_client)
        assert score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_score_via_llm_parse_error_falls_back_to_heuristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_via_llm falls back to heuristic on JSON parse error."""
        monitor = EngagementMonitor(profile="bill_hicks")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "not valid json {{"}
        mock_resp.raise_for_status = MagicMock()

        http_client = AsyncMock()
        http_client.post = AsyncMock(return_value=mock_resp)

        # "stop" triggers heuristic score of 1.0
        score = await monitor.score_via_llm("stop", http_client)
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_score_via_llm_network_error_falls_back_to_heuristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_via_llm falls back to heuristic on network error."""
        monitor = EngagementMonitor(profile="bill_hicks")

        http_client = AsyncMock()
        http_client.post = AsyncMock(side_effect=Exception("connection refused"))

        score = await monitor.score_via_llm("stop", http_client)
        assert score == pytest.approx(1.0)

    def test_analyze_with_llm_score_uses_llm_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """analyze() uses the provided llm_score when LLM scoring is enabled."""
        monkeypatch.setenv("REACHY_MINI_GUARDRAIL_LLM_SCORING", "1")
        monitor = EngagementMonitor(profile="bill_hicks")

        # Provide a high LLM score — should count as discomfort.
        score, _ = monitor.analyze("that's interesting", llm_score=0.9)
        assert score == pytest.approx(0.9)
        assert monitor.consecutive_discomfort == 1

    def test_analyze_without_llm_score_uses_heuristic_even_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """analyze() falls back to heuristic when llm_score is None."""
        monkeypatch.setenv("REACHY_MINI_GUARDRAIL_LLM_SCORING", "1")
        monitor = EngagementMonitor(profile="bill_hicks")

        # "stop" → heuristic 1.0
        score, _ = monitor.analyze("stop", llm_score=None)
        assert score == pytest.approx(1.0)

    def test_analyze_ignores_llm_score_when_feature_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LLM scoring is disabled, the provided llm_score is ignored."""
        monkeypatch.setenv("REACHY_MINI_GUARDRAIL_LLM_SCORING", "0")
        monitor = EngagementMonitor(profile="bill_hicks")

        # Heuristic → 0.0 for positive text; even if llm_score=0.95 is provided,
        # heuristic should win because the flag is off.
        score, _ = monitor.analyze(
            "That is a fascinating philosophical point.",
            llm_score=0.95,
        )
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Calibration logging
# ---------------------------------------------------------------------------


class TestCalibrationLogging:
    """Verify that analyze() emits the structured calibration DEBUG line."""

    def test_calibration_line_emitted_on_analyze(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        with caplog.at_level(logging.DEBUG, logger="robot_comic.guardrail"):
            monitor.analyze("stop")

        calibration_lines = [r for r in caplog.records if "guardrail.calibration" in r.message]
        assert len(calibration_lines) >= 1
        msg = calibration_lines[0].message
        assert "persona=" in msg
        assert "heuristic_score=" in msg
        assert "consecutive_discomfort=" in msg
        assert "should_soften=" in msg

    def test_calibration_line_contains_llm_score_null_when_not_provided(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        monitor = EngagementMonitor(profile="bill_hicks")
        with caplog.at_level(logging.DEBUG, logger="robot_comic.guardrail"):
            monitor.analyze("stop")

        calibration_lines = [r for r in caplog.records if "guardrail.calibration" in r.message]
        msg = calibration_lines[0].message
        assert "llm_score=null" in msg

    def test_calibration_line_contains_llm_score_when_provided(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REACHY_MINI_GUARDRAIL_LLM_SCORING", "1")
        monitor = EngagementMonitor(profile="bill_hicks")
        with caplog.at_level(logging.DEBUG, logger="robot_comic.guardrail"):
            monitor.analyze("stop", llm_score=0.75)

        calibration_lines = [r for r in caplog.records if "guardrail.calibration" in r.message]
        msg = calibration_lines[0].message
        assert "llm_score=0.75" in msg

    def test_calibration_line_emitted_for_disabled_monitor(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        monitor = EngagementMonitor(profile="don_rickles")
        with caplog.at_level(logging.DEBUG, logger="robot_comic.guardrail"):
            monitor.analyze("stop")

        calibration_lines = [r for r in caplog.records if "guardrail.calibration" in r.message]
        assert len(calibration_lines) >= 1


# ---------------------------------------------------------------------------
# SOFTEN_NOTE constant (backwards compat)
# ---------------------------------------------------------------------------


def test_soften_note_is_non_empty() -> None:
    assert SOFTEN_NOTE and len(SOFTEN_NOTE) > 20
