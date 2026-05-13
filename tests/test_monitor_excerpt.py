"""Tests for pending-row transcript excerpt display in robot-comic-monitor.

Covers the fix for issue #93: the monitor should show the transcript excerpt on
the in-flight (pending) row by reading it from the closed ``stt.infer`` child
span while the outer ``turn`` span is still open.
"""

from robot_comic.monitor import SpanBuffer, _build_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stt_infer_span(trace_id: str, excerpt: str = "", dur_ms: float = 250.0) -> dict:
    """Build a minimal closed stt.infer span dict as emitted by CompactLineExporter."""
    attrs: dict = {}
    if excerpt:
        attrs["turn.excerpt"] = excerpt
    return {
        "name": "stt.infer",
        "trace": trace_id,
        "span": "aabbccdd11223344",
        "parent": "0011223344556677",  # non-None → not a root turn span
        "dur_ms": dur_ms,
        "status": "OK",
        "ts": 1_700_000_000_000,
        "attrs": attrs,
    }


def _turn_span(trace_id: str, excerpt: str = "") -> dict:
    """Build a root turn span (closes the trace)."""
    attrs: dict = {"robot.mode": "local_stt"}
    if excerpt:
        attrs["turn.excerpt"] = excerpt
    return {
        "name": "turn",
        "trace": trace_id,
        "span": "0011223344556677",
        "parent": None,
        "dur_ms": 1500.0,
        "status": "OK",
        "ts": 1_700_000_001_500,
        "attrs": attrs,
    }


# ---------------------------------------------------------------------------
# SpanBuffer data-extraction tests
# ---------------------------------------------------------------------------


def test_pending_excerpt_from_stt_infer_span() -> None:
    """SpanBuffer picks up turn.excerpt from a closed stt.infer child span."""
    buf = SpanBuffer()
    tid = "aabbccdd" * 4  # 32 hex chars

    result = buf.ingest(_stt_infer_span(tid, excerpt="tell me a quick joke…"))
    assert result is None, "stt.infer should not complete the turn"

    pending = buf.latest_pending
    assert pending is not None
    assert pending.excerpt == "tell me a quick joke…"


def test_pending_excerpt_empty_when_no_stt_infer_attrs() -> None:
    """PendingTurn excerpt stays empty when stt.infer carries no turn.excerpt attr."""
    buf = SpanBuffer()
    tid = "11223344" * 4

    buf.ingest(_stt_infer_span(tid, excerpt=""))  # no excerpt attribute

    pending = buf.latest_pending
    assert pending is not None
    assert pending.excerpt == ""


def test_pending_excerpt_cleared_on_turn_completion() -> None:
    """Once the outer turn span closes, the pending turn is removed from inflight."""
    buf = SpanBuffer()
    tid = "99aabbcc" * 4

    buf.ingest(_stt_infer_span(tid, excerpt="hello robot"))
    turn_record = buf.ingest(_turn_span(tid))

    assert turn_record is not None, "Root turn span should complete the TurnRecord"
    assert buf.latest_pending is None, "No pending turn should remain after completion"


# ---------------------------------------------------------------------------
# Rendering tests
# ---------------------------------------------------------------------------


def test_build_table_pending_row_shows_excerpt() -> None:
    """_build_table renders the excerpt in the What column when available."""
    buf = SpanBuffer()
    tid = "deadbeef" * 4

    buf.ingest(_stt_infer_span(tid, excerpt="what is your name…"))
    pending = buf.latest_pending
    assert pending is not None

    table = _build_table([], pending)

    # Collect all plain-text values rendered in the table
    rendered_texts = []
    for col in table.columns:
        for cell in col._cells:
            if hasattr(cell, "plain"):
                rendered_texts.append(cell.plain)
            elif isinstance(cell, str):
                rendered_texts.append(cell)

    assert any("what is your name" in t for t in rendered_texts), (
        f"Expected excerpt in pending row cells, got: {rendered_texts}"
    )


def test_build_table_pending_row_shows_spinner_when_no_excerpt() -> None:
    """_build_table still renders a spinner when the pending turn has no excerpt."""
    buf = SpanBuffer()
    tid = "cafebabe" * 4

    buf.ingest(_stt_infer_span(tid, excerpt=""))
    pending = buf.latest_pending
    assert pending is not None
    assert pending.excerpt == ""

    table = _build_table([], pending)

    # The What column (index 1) should contain a spinner character, not empty
    what_cells = list(table.columns[1]._cells)
    assert what_cells, "Expected at least one cell in the What column"
    what_plain = what_cells[0].plain if hasattr(what_cells[0], "plain") else str(what_cells[0])
    # Spinner chars are single Unicode braille characters; just ensure non-empty
    assert what_plain.strip(), "Spinner cell should be non-empty when no excerpt is available"
