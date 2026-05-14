"""Unit tests for the ``robot_comic.tools.name_validation`` helper (#287)."""

from __future__ import annotations
import logging

import pytest

from robot_comic.tools.name_validation import (
    RECENT_USER_TRANSCRIPTS_MAXLEN,
    name_in_transcripts,
    validate_name_or_warn,
    record_user_transcript,
)


# ── name_in_transcripts ────────────────────────────────────────────────────────


def test_name_in_transcripts_word_boundary_match() -> None:
    assert name_in_transcripts("Tony", ["Hi I'm Tony"])


def test_name_in_transcripts_case_insensitive() -> None:
    assert name_in_transcripts("Tony", ["my name is tony"])
    assert name_in_transcripts("tony", ["Hi I'm TONY"])


def test_name_in_transcripts_rejects_substring() -> None:
    # Word boundary defends against "Anton" matching "Antonio".
    assert not name_in_transcripts("Anton", ["I'm Antonio from Rome"])


def test_name_in_transcripts_rejects_when_absent() -> None:
    assert not name_in_transcripts("John", ["Hello", "What's up"])


def test_name_in_transcripts_empty_inputs() -> None:
    assert not name_in_transcripts("", ["Tony"])
    assert not name_in_transcripts("   ", ["Tony"])
    assert not name_in_transcripts("Tony", [])
    assert not name_in_transcripts("Tony", [""])


def test_name_in_transcripts_regex_special_chars_are_escaped() -> None:
    # The name "C-3PO" contains regex meta-characters that re.escape neutralises.
    assert name_in_transcripts("C-3PO", ["I'm C-3PO, human-cyborg relations"])
    assert not name_in_transcripts("C-3PO", ["I'm Threepio"])


# ── record_user_transcript ─────────────────────────────────────────────────────


def test_record_user_transcript_appends() -> None:
    buf: list[str] = []
    record_user_transcript(buf, "hello")
    record_user_transcript(buf, "world")
    assert buf == ["hello", "world"]


def test_record_user_transcript_strips_and_skips_empty() -> None:
    buf: list[str] = []
    record_user_transcript(buf, "")
    record_user_transcript(buf, "   ")
    record_user_transcript(buf, None)
    record_user_transcript(buf, "  hi  ")
    assert buf == ["hi"]


def test_record_user_transcript_bounded_to_maxlen() -> None:
    buf: list[str] = []
    for i in range(RECENT_USER_TRANSCRIPTS_MAXLEN + 3):
        record_user_transcript(buf, f"turn{i}")
    assert len(buf) == RECENT_USER_TRANSCRIPTS_MAXLEN
    # Oldest entries dropped first; most recent is last.
    assert buf[-1] == f"turn{RECENT_USER_TRANSCRIPTS_MAXLEN + 2}"


# ── validate_name_or_warn ──────────────────────────────────────────────────────


def test_validate_name_or_warn_returns_true_for_real_name() -> None:
    assert validate_name_or_warn("Tony", ["I'm Tony"], tool_name="greet.identify")


def test_validate_name_or_warn_logs_warning_with_forensic_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transcripts = ["Hello there", "How are you"]
    with caplog.at_level(logging.WARNING):
        ok = validate_name_or_warn("John", transcripts, tool_name="greet.identify")
    assert ok is False
    assert any(
        "rejected name" in r.message and "John" in r.message and "greet.identify" in r.message for r in caplog.records
    )
