"""Disengagement guardrail for high-intensity personas (Bill Hicks).

``EngagementMonitor`` inspects each user turn for discomfort signals using a
lightweight keyword/regex heuristic — no LLM call, no network, O(n) on the
phrase list.

Activation is persona-aware: only profiles listed in
``GUARDRAIL_PROFILES`` (default: ``{"bill_hicks"}``) activate the monitor.
The feature can be force-disabled for any profile via the
``REACHY_MINI_GUARDRAIL_ENABLED`` environment variable.

Soften note injected into the next system prompt when ``should_soften`` is True::

    NOTE: the user seems uncomfortable with the current intensity — pivot to
    lighter, observational material for the next few turns.
"""

from __future__ import annotations
import os
import re
import logging


logger = logging.getLogger(__name__)

# The single-line note prepended (at runtime only) to the system prompt when
# the monitor decides the persona should soften.  Not persisted anywhere.
SOFTEN_NOTE = (
    "NOTE: the user seems uncomfortable with the current intensity — "
    "pivot to lighter, observational material for the next few turns."
)

# Profiles that opt in to the guardrail by default.
GUARDRAIL_PROFILES: frozenset[str] = frozenset({"bill_hicks"})

# Number of consecutive discomfort turns required before ``should_soften``
# fires.  Override via ``REACHY_MINI_GUARDRAIL_THRESHOLD``.
_DEFAULT_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Discomfort signal patterns
# ---------------------------------------------------------------------------
# Phrases that indicate the user wants to disengage or tone things down.
# Each entry is compiled into a case-insensitive regex with word-boundary
# anchors where helpful.  The list is intentionally conservative to avoid
# false positives on content-level disagreement (which Hicks *should* engage).
_DISCOMFORT_PHRASES: list[str] = [
    r"\bstop\b",
    r"\benough\b",
    r"\btone.?it.?down\b",
    r"\bless.?aggressive\b",
    r"\bchange.?the.?topic\b",
    r"\bchange.?subject\b",
    r"\blet.?s.?change\b",
    r"\bokay.?enough\b",
    r"\bthat.?s.?enough\b",
    r"\btoo.?much\b",
    r"\bback.?off\b",
    r"\bease.?up\b",
    r"\bcalm.?down\b",
    r"\bnot.?funny\b",
    r"\bnot.?comfortable\b",
    r"\buncomfortable\b",
    r"\bplease.?stop\b",
    r"\bi.?don.?t.?like.?this\b",
    r"\bi.?don.?t.?want.?to\b",
    r"\bcan.?we.?talk.?about.?something.?else\b",
    r"\bcan.?we.?change\b",
    r"\bnever.?mind\b",
    r"\bforget.?it\b",
    r"\bquit.?it\b",
    r"\bjust.?stop\b",
    r"\bwhatever\b",
    r"\bi.?give.?up\b",
    r"\bleave.?me.?alone\b",
    r"\bgo.?away\b",
]

# Compile once at module load.
_DISCOMFORT_RE: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in _DISCOMFORT_PHRASES
]

# Minimum word count below which a response is treated as "very short"
# (a mild secondary signal — not enough alone, raises score by less).
_SHORT_RESPONSE_WORDS = 3
_SHORT_RESPONSE_SCORE_CONTRIBUTION = 0.3


def _score_discomfort(text: str) -> float:
    """Return a discomfort score in ``[0.0, 1.0]`` for a single user turn.

    Scoring rules:
    - Empty / blank input → 0.6 (silence is a strong disengagement signal).
    - Any discomfort phrase match → 1.0 (hard signal; return immediately).
    - Very short response (≤ ``_SHORT_RESPONSE_WORDS`` words) without a phrase
      match → ``_SHORT_RESPONSE_SCORE_CONTRIBUTION`` (weak signal).
    - Otherwise → 0.0.
    """
    stripped = text.strip()
    if not stripped:
        return 0.6

    for pattern in _DISCOMFORT_RE:
        if pattern.search(stripped):
            return 1.0

    words = stripped.split()
    if len(words) <= _SHORT_RESPONSE_WORDS:
        return _SHORT_RESPONSE_SCORE_CONTRIBUTION

    return 0.0


def _guardrail_enabled_for_profile(profile: str | None) -> bool:
    """Return whether the guardrail is active for the given profile name.

    Resolution order:
    1. ``REACHY_MINI_GUARDRAIL_ENABLED`` env var (explicit override, any profile).
    2. Profile membership in ``GUARDRAIL_PROFILES``.
    """
    raw = os.getenv("REACHY_MINI_GUARDRAIL_ENABLED", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # No explicit override — use profile membership.
    return (profile or "") in GUARDRAIL_PROFILES


def _get_threshold() -> int:
    """Return the consecutive-discomfort threshold from env or default."""
    raw = os.getenv("REACHY_MINI_GUARDRAIL_THRESHOLD", "").strip()
    if raw:
        try:
            val = int(raw)
            if val >= 1:
                return val
            logger.warning("REACHY_MINI_GUARDRAIL_THRESHOLD=%r must be >= 1; using default %d", raw, _DEFAULT_THRESHOLD)
        except ValueError:
            logger.warning("REACHY_MINI_GUARDRAIL_THRESHOLD=%r is not an integer; using default %d", raw, _DEFAULT_THRESHOLD)
    return _DEFAULT_THRESHOLD


class EngagementMonitor:
    r"""Stateful per-session monitor for user disengagement signals.

    Typical lifecycle::

        monitor = EngagementMonitor(profile="bill_hicks")
        score, should_soften = monitor.analyze(user_text)
        if should_soften:
            instructions = SOFTEN_NOTE + "\n\n" + base_instructions

    The monitor is a no-op (``score=0.0, should_soften=False``) when the
    active profile is not in ``GUARDRAIL_PROFILES`` and no env-override is set.
    """

    def __init__(self, profile: str | None = None) -> None:
        """Initialise the monitor for the given profile (or no profile)."""
        self._profile = profile
        self._enabled = _guardrail_enabled_for_profile(profile)
        self._threshold = _get_threshold()
        self._consecutive_discomfort: int = 0
        self._last_score: float = 0.0
        logger.debug(
            "EngagementMonitor init: profile=%r enabled=%s threshold=%d",
            profile,
            self._enabled,
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return whether the guardrail is active for this monitor's profile."""
        return self._enabled

    @property
    def consecutive_discomfort(self) -> int:
        """Number of consecutive turns with a discomfort score above threshold."""
        return self._consecutive_discomfort

    @property
    def last_score(self) -> float:
        """Discomfort score from the most recent ``analyze()`` call."""
        return self._last_score

    def analyze(self, user_text: str) -> tuple[float, bool]:
        """Inspect *user_text* and update engagement state.

        Args:
            user_text: The raw transcript of the latest user turn.

        Returns:
            ``(discomfort_score, should_soften)`` where:
            - ``discomfort_score`` is in ``[0.0, 1.0]``.
            - ``should_soften`` is ``True`` when the monitor recommends that
              the persona ease back for the next few turns.

        """
        if not self._enabled:
            return 0.0, False

        score = _score_discomfort(user_text)
        self._last_score = score

        # A score ≥ 0.5 counts as a discomfort signal.
        if score >= 0.5:
            self._consecutive_discomfort += 1
            logger.debug(
                "EngagementMonitor: discomfort score=%.2f consecutive=%d threshold=%d",
                score,
                self._consecutive_discomfort,
                self._threshold,
            )
        else:
            if self._consecutive_discomfort > 0:
                logger.debug(
                    "EngagementMonitor: engagement signal (score=%.2f) — resetting consecutive counter",
                    score,
                )
            self._consecutive_discomfort = 0

        should_soften = self._consecutive_discomfort >= self._threshold
        if should_soften:
            logger.info(
                "EngagementMonitor: soften triggered (consecutive=%d >= threshold=%d)",
                self._consecutive_discomfort,
                self._threshold,
            )
        return score, should_soften

    def reset(self) -> None:
        """Reset engagement state (e.g., on new session or profile switch)."""
        self._consecutive_discomfort = 0
        self._last_score = 0.0
        logger.debug("EngagementMonitor: state reset for profile=%r", self._profile)

    def update_profile(self, profile: str | None) -> None:
        """Update the active profile and re-evaluate whether the guardrail is enabled."""
        self._profile = profile
        self._enabled = _guardrail_enabled_for_profile(profile)
        self.reset()
        logger.debug(
            "EngagementMonitor: profile updated to %r, enabled=%s",
            profile,
            self._enabled,
        )
