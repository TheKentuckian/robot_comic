"""Tests for supporting-event rows in robot-comic-monitor (#301).

Supporting events are top-level spans carrying ``event.kind=supporting`` that
surface boot-timeline checkpoints (``app.startup``, ``welcome.wav.played``,
``handler.start_up.complete``, ``first_greeting.tts_first_audio`` …) on the
same chronological lane as full conversational turns.
"""

from __future__ import annotations
from typing import Any

from robot_comic.monitor import SpanBuffer, SupportingEvent, _build_table


# ---------------------------------------------------------------------------
# Span fixture helpers
# ---------------------------------------------------------------------------


def _supporting_span(
    name: str,
    *,
    trace_id: str = "00" * 16,
    ts_ms: int = 1_700_000_000_000,
    dur_ms: float = 0.0,
    event_dur_ms: float | None = None,
) -> dict[str, Any]:
    """Build a supporting-event span dict as emitted by CompactLineExporter."""
    attrs: dict[str, Any] = {"event.kind": "supporting"}
    if event_dur_ms is not None:
        attrs["event.dur_ms"] = float(event_dur_ms)
    return {
        "name": name,
        "trace": trace_id,
        "span": "aabbccdd11223344",
        "parent": None,
        "dur_ms": dur_ms,
        "status": "OK",
        "ts": ts_ms,
        "attrs": attrs,
    }


def _turn_span(
    trace_id: str,
    *,
    ts_ms: int = 1_700_000_010_000,
    dur_ms: float = 1500.0,
    excerpt: str = "hello",
) -> dict[str, Any]:
    """Build a root turn span (closes the trace)."""
    return {
        "name": "turn",
        "trace": trace_id,
        "span": "0011223344556677",
        "parent": None,
        "dur_ms": dur_ms,
        "status": "OK",
        "ts": ts_ms,
        "attrs": {
            "robot.mode": "openai_realtime",
            "turn.outcome": "success",
            "turn.excerpt": excerpt,
        },
    }


# ---------------------------------------------------------------------------
# SpanBuffer routing
# ---------------------------------------------------------------------------


def test_span_buffer_routes_supporting_to_separate_list() -> None:
    """Supporting spans go to ``supporting_events``, not the turn pipeline."""
    buf = SpanBuffer()

    result = buf.ingest(_supporting_span("app.startup", event_dur_ms=2100.0))

    assert result is None, "Supporting events never complete a TurnRecord"
    assert len(buf.supporting_events) == 1
    ev = buf.supporting_events[0]
    assert ev.name == "app.startup"
    assert ev.dur_ms == 2100.0


def test_span_buffer_prefers_event_dur_ms_over_span_dur_ms() -> None:
    """``event.dur_ms`` overrides the span's own wall-clock when present."""
    buf = SpanBuffer()
    buf.ingest(_supporting_span("welcome.wav.played", dur_ms=0.1, event_dur_ms=1800.0))
    assert buf.supporting_events[0].dur_ms == 1800.0


def test_span_buffer_falls_back_to_span_dur_ms_when_event_dur_missing() -> None:
    """Without ``event.dur_ms`` the span's own wall-clock is used."""
    buf = SpanBuffer()
    buf.ingest(_supporting_span("some.event", dur_ms=42.5))
    assert buf.supporting_events[0].dur_ms == 42.5


def test_span_buffer_does_not_affect_turn_pipeline() -> None:
    """A supporting span does not interfere with an in-flight turn."""
    buf = SpanBuffer()
    tid = "deadbeef" * 4

    buf.ingest(_supporting_span("app.startup", event_dur_ms=1000.0))
    turn = buf.ingest(_turn_span(tid))

    assert turn is not None, "Turn span should still produce a TurnRecord"
    assert len(buf.supporting_events) == 1


# ---------------------------------------------------------------------------
# Render — chronological interleaving
# ---------------------------------------------------------------------------


def _row_labels(table: Any) -> list[str]:
    """Return the plain-text contents of column 1 (the What/kind column)."""
    cells = list(table.columns[1]._cells)
    out: list[str] = []
    for cell in cells:
        if hasattr(cell, "plain"):
            out.append(cell.plain)
        else:
            out.append(str(cell))
    return out


def test_render_interleaves_supporting_and_turn_rows_chronologically() -> None:
    """A mix of supporting and turn spans is rendered newest-first by ts."""
    buf = SpanBuffer()

    # Boot timeline (older) — issue #301 sample ordering
    buf.ingest(_supporting_span("app.startup", ts_ms=1_700_000_000_000, event_dur_ms=2100.0))
    buf.ingest(_supporting_span("welcome.wav.played", ts_ms=1_700_000_002_000, event_dur_ms=1800.0))
    buf.ingest(_supporting_span("handler.start_up.complete", ts_ms=1_700_000_028_000, event_dur_ms=28500.0))
    buf.ingest(_supporting_span("first_greeting.tts_first_audio", ts_ms=1_700_000_042_000, event_dur_ms=42000.0))
    # Subsequent conversational turns
    turn1 = buf.ingest(_turn_span("a" * 32, ts_ms=1_700_000_048_000, excerpt="greeting"))
    turn2 = buf.ingest(_turn_span("b" * 32, ts_ms=1_700_000_090_000, excerpt="Hello"))

    assert turn1 is not None and turn2 is not None
    turns = [turn1, turn2]

    table = _build_table(turns, pending=None, supporting_events=buf.supporting_events)
    labels = _row_labels(table)

    # Newest first → reverse chronological. Six rows total.
    assert labels == [
        "Hello",
        "greeting",
        "first_greeting.tts_first_audio",
        "handler.start_up.complete",
        "welcome.wav.played",
        "app.startup",
    ], labels


def test_render_supporting_row_blank_stage_columns() -> None:
    """Supporting rows leave STT/LLM/TTS columns blank, only Total populated."""
    buf = SpanBuffer()
    buf.ingest(_supporting_span("app.startup", ts_ms=1_700_000_000_000, event_dur_ms=2100.0))

    table = _build_table([], pending=None, supporting_events=buf.supporting_events)

    # Column indices: 0=Time, 1=What, 2=STT, 3=LLM, 4=TTS, 5=Total, 6=Tools, 7=icon
    def _first(col_idx: int) -> str:
        cell = list(table.columns[col_idx]._cells)[0]
        return cell.plain if hasattr(cell, "plain") else str(cell)

    assert _first(1) == "app.startup"
    assert _first(2).strip() == ""  # STT blank
    assert _first(3).strip() == ""  # LLM blank
    assert _first(4).strip() == ""  # TTS blank
    # Total carries the formatted duration (2.1 s).
    assert "2.1s" in _first(5) or "2100ms" in _first(5)


def test_render_only_supporting_events_no_turns() -> None:
    """A table built from only supporting events renders them all."""
    buf = SpanBuffer()
    buf.ingest(_supporting_span("app.startup", ts_ms=1_700_000_000_000, event_dur_ms=2100.0))
    buf.ingest(_supporting_span("welcome.wav.played", ts_ms=1_700_000_002_000, event_dur_ms=1800.0))

    table = _build_table([], pending=None, supporting_events=buf.supporting_events)
    labels = _row_labels(table)

    assert labels == ["welcome.wav.played", "app.startup"]


# ---------------------------------------------------------------------------
# End-to-end: parse RCSPAN lines, build table
# ---------------------------------------------------------------------------


def test_end_to_end_rcspan_lines_yield_interleaved_rows() -> None:
    """Feed RCSPAN-formatted lines through _parse_span + SpanBuffer + _build_table."""
    import json

    from robot_comic.monitor import _parse_span

    spans = [
        _supporting_span("app.startup", ts_ms=1_700_000_000_000, event_dur_ms=2100.0),
        _supporting_span("daemon.wake_up", ts_ms=1_700_000_001_000, event_dur_ms=2200.0),
        _supporting_span("welcome.wav.played", ts_ms=1_700_000_003_000, event_dur_ms=1800.0),
        _supporting_span(
            "handler.start_up.complete",
            ts_ms=1_700_000_030_000,
            event_dur_ms=28500.0,
        ),
        _supporting_span(
            "first_greeting.tts_first_audio",
            ts_ms=1_700_000_032_000,
            event_dur_ms=32000.0,
        ),
        _turn_span("a" * 32, ts_ms=1_700_000_035_000, excerpt="greeting"),
        _turn_span("b" * 32, ts_ms=1_700_000_080_000, excerpt="Hello"),
    ]

    buf = SpanBuffer()
    turns = []
    for sp in spans:
        line = f"RCSPAN {json.dumps(sp)}"
        parsed = _parse_span(line)
        assert parsed is not None
        result = buf.ingest(parsed)
        if result is not None:
            turns.append(result)

    assert len(turns) == 2
    assert len(buf.supporting_events) == 5

    table = _build_table(turns, pending=None, supporting_events=buf.supporting_events)
    labels = _row_labels(table)

    # All 7 rows present, newest first.
    assert labels == [
        "Hello",
        "greeting",
        "first_greeting.tts_first_audio",
        "handler.start_up.complete",
        "welcome.wav.played",
        "daemon.wake_up",
        "app.startup",
    ]


# ---------------------------------------------------------------------------
# SupportingEvent dataclass-style construction
# ---------------------------------------------------------------------------


def test_supporting_event_handles_invalid_event_dur_ms() -> None:
    """A non-numeric ``event.dur_ms`` falls back to the span's own ``dur_ms``."""
    span = _supporting_span("x", dur_ms=12.0)
    span["attrs"]["event.dur_ms"] = "not a number"
    ev = SupportingEvent(span)
    assert ev.dur_ms == 12.0
