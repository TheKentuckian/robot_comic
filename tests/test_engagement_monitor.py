"""Unit tests for EngagementMonitor (guardrail.py).

Pure unit tests — no LLM, no audio, no network.
"""

import pytest

from robot_comic.guardrail import SOFTEN_NOTE, GUARDRAIL_PROFILES, EngagementMonitor


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
        score, _ = monitor.analyze(
            "That's a fascinating point. I never thought about it that way before."
        )
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

    def test_other_profile_disabled_by_default(self) -> None:
        monitor = EngagementMonitor(profile="don_rickles")
        assert monitor.enabled is False

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
# SOFTEN_NOTE constant
# ---------------------------------------------------------------------------


def test_soften_note_is_non_empty() -> None:
    assert SOFTEN_NOTE and len(SOFTEN_NOTE) > 20
