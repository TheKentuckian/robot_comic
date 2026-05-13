"""Live TUI monitor for robot_comic turn traces.

Tails the systemd journal (or a log file) for RCSPAN lines emitted by
CompactLineExporter and renders a rolling table of completed turns plus
a live in-flight row that updates as each stage (STT→LLM→TTS) finishes.

Usage:
    robot-comic-monitor                          # tail reachy-app-autostart unit
    robot-comic-monitor --unit my-service        # different systemd unit
    robot-comic-monitor --file /path/to/app.log  # tail a file instead
"""

import sys
import json
import time
import argparse
import threading
import subprocess
from typing import Any, Iterator, Optional
from datetime import datetime
from collections import defaultdict


try:
    from rich import box
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
except ImportError:
    print("rich is required: uv pip install 'robot_comic[monitor]'", file=sys.stderr)
    raise SystemExit(1)


_JOURNALD_UNIT = "reachy-app-autostart"
_MAX_TURNS = 30

# Thresholds (ms) for green / yellow / red colouring per stage
_THRESHOLDS: dict[str, tuple[float, float]] = {
    "stt": (300, 600),
    "llm": (800, 2000),
    "tts": (500, 1500),
    "total": (2000, 4000),
}

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spin() -> str:
    return _SPINNER[int(time.time() * 8) % len(_SPINNER)]


def _service_active(unit: str) -> bool:
    """Return True iff the given systemd unit is currently active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            timeout=1.0,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TurnRecord:
    """Aggregates all spans for one conversational turn."""

    def __init__(self, trace_id: str, root: dict[str, Any], children: list[dict[str, Any]]) -> None:
        """Build a TurnRecord from the root turn span and its buffered children."""
        self.trace_id = trace_id
        self.root = root
        self.children = children
        self.ts = datetime.fromtimestamp(root["ts"] / 1000)

    def _kids(self, name: str) -> list[dict[str, Any]]:
        return [s for s in self.children if s["name"] == name]

    def _sum_ms(self, name: str) -> Optional[float]:
        kids = self._kids(name)
        return sum(s["dur_ms"] for s in kids) if kids else None

    @property
    def stt_ms(self) -> Optional[float]:
        """Total STT inference time."""
        return self._sum_ms("stt.infer")

    @property
    def llm_ms(self) -> Optional[float]:
        """Total LLM time (summed across tool-call chains)."""
        return self._sum_ms("llm.request")

    @property
    def tts_ms(self) -> Optional[float]:
        """Total TTS synthesis time."""
        return self._sum_ms("tts.synthesize")

    @property
    def total_ms(self) -> float:
        """End-to-end turn duration."""
        return float(self.root["dur_ms"])

    @property
    def outcome(self) -> str:
        """Turn outcome attribute."""
        return str(self.root["attrs"].get("turn.outcome", "?"))

    @property
    def mode(self) -> str:
        """Backend mode (openai, chatterbox, gemini, etc.)."""
        return str(self.root["attrs"].get("robot.mode", "?"))

    @property
    def excerpt(self) -> str:
        """Short label: first few words of transcript, or 'greeting' for startup."""
        val = self.root["attrs"].get("turn.excerpt", "")
        if val:
            return str(val)
        return ""

    @property
    def tool_count(self) -> int:
        """Number of tool calls made during this turn."""
        return len(self._kids("tool.execute"))


class PendingTurn:
    """Tracks a turn that has started (stt.infer seen) but not yet completed."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.ts = datetime.now()
        self.started_at = time.perf_counter()
        self.stt_ms: Optional[float] = None
        self.llm_ms: float = 0.0
        self.llm_count: int = 0  # completed llm.request spans seen
        self.tts_ms: float = 0.0
        self.tts_count: int = 0  # completed tts.synthesize spans seen
        self.tool_count: int = 0
        self.excerpt: str = ""  # populated from stt.infer turn.excerpt attribute

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000


# ---------------------------------------------------------------------------
# Span buffer
# ---------------------------------------------------------------------------


class SpanBuffer:
    """Accumulates child spans until the root turn span closes the trace."""

    def __init__(self) -> None:
        """Initialise the pending-span store."""
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._inflight: dict[str, PendingTurn] = {}

    def ingest(self, span: dict[str, Any]) -> Optional[TurnRecord]:
        """Return a completed TurnRecord when the root turn span arrives, else None."""
        trace_id = span["trace"]
        name = span["name"]

        if name == "turn" and span["parent"] is None:
            children = self._pending.pop(trace_id, [])
            self._inflight.pop(trace_id, None)
            return TurnRecord(trace_id, span, children)

        self._pending[trace_id].append(span)

        # Update in-flight tracking so the pending row reflects current progress.
        if name == "stt.infer":
            if trace_id not in self._inflight:
                self._inflight[trace_id] = PendingTurn(trace_id)
            self._inflight[trace_id].stt_ms = span.get("dur_ms")
            # Capture transcript excerpt from the completed stt.infer span so the
            # pending row can display it while the outer turn span is still open.
            excerpt = span.get("attrs", {}).get("turn.excerpt", "")
            if excerpt:
                self._inflight[trace_id].excerpt = str(excerpt)
        elif name == "llm.request":
            pt = self._inflight.get(trace_id)
            if pt is not None:
                pt.llm_ms += span.get("dur_ms", 0.0)
                pt.llm_count += 1
        elif name == "tts.synthesize":
            pt = self._inflight.get(trace_id)
            if pt is not None:
                pt.tts_ms += span.get("dur_ms", 0.0)
                pt.tts_count += 1
        elif name == "tool.execute":
            pt = self._inflight.get(trace_id)
            if pt is not None:
                pt.tool_count += 1

        return None

    @property
    def latest_pending(self) -> Optional[PendingTurn]:
        """Return the most-recently-started in-flight turn, if any."""
        if not self._inflight:
            return None
        return max(self._inflight.values(), key=lambda pt: pt.started_at)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(ms: Optional[float], stage: str) -> Text:
    """Format a millisecond value with threshold-based colour."""
    if ms is None:
        return Text("  —  ", style="dim")
    warn, bad = _THRESHOLDS.get(stage, (500, 2000))
    if ms >= bad:
        style = "bold red"
    elif ms >= warn:
        style = "yellow"
    else:
        style = "green"
    label = f"{ms:.0f}ms" if ms < 10_000 else f"{ms / 1000:.1f}s"
    return Text(label.rjust(7), style=style)


def _fmt_spin(ms: Optional[float], active: bool, stage: str) -> Text:
    """Format a stage cell for the in-flight pending row.

    active=True  → stage has started (show spinner while waiting, time when done)
    active=False → stage not yet started (show dim dash)
    """
    if ms is not None and ms > 0:
        return _fmt(ms, stage)
    if active:
        return Text(f"  {_spin()}    ", style="yellow")
    return Text("  —  ", style="dim")


def _build_table(turns: list[TurnRecord], pending: Optional[PendingTurn] = None) -> Table:
    """Render the most recent turns as a Rich table (newest row first)."""
    t = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style="bold dim",
        row_styles=["", "dim"],
    )
    t.add_column("Time", width=8, no_wrap=True)
    t.add_column("What", width=20, no_wrap=True)
    t.add_column("STT", width=8, justify="right")
    t.add_column("LLM", width=8, justify="right")
    t.add_column("TTS", width=8, justify="right")
    t.add_column("Total", width=8, justify="right")
    t.add_column("Tools", width=5, justify="right")
    t.add_column("", width=1, justify="center")

    if pending is not None:
        sp = _spin()
        stt_done = pending.stt_ms is not None
        llm_started = stt_done  # LLM starts right after STT
        tts_started = pending.llm_count > 0
        # LLM spinner: active while STT is done but no llm.request seen yet,
        # or after the last llm request if tts hasn't started.
        llm_active = llm_started and pending.llm_count == 0
        # Show accumulated LLM time if any requests completed.
        llm_ms = pending.llm_ms if pending.llm_count > 0 else None
        tts_active = tts_started and pending.tts_count == 0
        tts_ms = pending.tts_ms if pending.tts_count > 0 else None

        # Show excerpt from stt.infer span if available, otherwise spinner.
        what_cell: Text
        if pending.excerpt:
            what_cell = Text(pending.excerpt, style="italic yellow")
        else:
            what_cell = Text(f"{sp}", style="bold yellow")
        t.add_row(
            pending.ts.strftime("%H:%M:%S"),
            what_cell,
            _fmt_spin(pending.stt_ms, stt_done, "stt"),
            _fmt_spin(llm_ms, llm_active, "llm"),
            _fmt_spin(tts_ms, tts_active, "tts"),
            _fmt(pending.elapsed_ms, "total"),
            Text(str(pending.tool_count), style="cyan") if pending.tool_count else Text(""),
            Text(sp, style="yellow"),
        )

    for turn in reversed(turns[-_MAX_TURNS:]):
        ok_icon = Text("✓", style="green") if turn.outcome == "success" else Text("✗", style="red")
        tools_cell = Text(str(turn.tool_count), style="cyan") if turn.tool_count else Text("")
        excerpt = turn.excerpt or turn.mode
        t.add_row(
            turn.ts.strftime("%H:%M:%S"),
            Text(excerpt, style="italic dim" if excerpt == "greeting" else ""),
            _fmt(turn.stt_ms, "stt"),
            _fmt(turn.llm_ms, "llm"),
            _fmt(turn.tts_ms, "tts"),
            _fmt(turn.total_ms, "total"),
            tools_cell,
            ok_icon,
        )
    return t


# ---------------------------------------------------------------------------
# Log tailing
# ---------------------------------------------------------------------------


def _iter_journald(unit: str) -> Iterator[str]:
    """Stream lines from the systemd journal for a given unit."""
    proc = subprocess.Popen(
        ["journalctl", "-u", unit, "-f", "-o", "cat", "--no-pager"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip()
    finally:
        proc.terminate()


def _iter_file(path: str) -> Iterator[str]:
    """Tail a log file, yielding new lines as they appear."""
    with open(path) as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip()
            else:
                time.sleep(0.05)


def _parse_span(line: str) -> Optional[dict[str, Any]]:
    """Extract a span dict from a line containing an RCSPAN JSON payload."""
    idx = line.find("RCSPAN ")
    if idx == -1:
        return None
    try:
        result: dict[str, Any] = json.loads(line[idx + 7 :])
        return result
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the live TUI monitor."""
    parser = argparse.ArgumentParser(
        description="Live TUI monitor for robot_comic turn traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--unit",
        default=_JOURNALD_UNIT,
        metavar="UNIT",
        help=f"systemd unit to tail (default: {_JOURNALD_UNIT})",
    )
    source.add_argument(
        "--file",
        metavar="PATH",
        help="tail a log file instead of the systemd journal",
    )
    args = parser.parse_args()

    console = Console()
    buffer = SpanBuffer()
    turns: list[TurnRecord] = []

    lines: Iterator[str] = _iter_file(args.file) if args.file else _iter_journald(args.unit)
    source_label = f"file:{args.file}" if args.file else f"journald:{args.unit}"
    # Polling unit name for service status indicator (only relevant for journald mode).
    watch_unit: Optional[str] = args.unit if not args.file else None

    # Service status — polled every ~3 s by the refresh thread.
    _svc_active: list[bool] = [True]  # mutable box so refresh thread can update it
    _svc_last_check: list[float] = [0.0]

    def _render() -> Panel:
        pending = buffer.latest_pending

        # Service status indicator
        if watch_unit is not None:
            dot = "● " if _svc_active[0] else "○ "
            dot_style = "green" if _svc_active[0] else "red"
            svc_text = Text(dot, style=dot_style)
            svc_text.append(watch_unit, style="bold" if _svc_active[0] else "dim")
            title_suffix = f"  [dim]{source_label}[/dim]"
        else:
            svc_text = None
            title_suffix = f"  [dim]{source_label}[/dim]"

        body: Text | Table
        if not turns and pending is None:
            if watch_unit is not None and not _svc_active[0]:
                body = Text("Service is stopped — start it to see turns.", style="dim italic")
            else:
                body = Text("Waiting for turns… (is ROBOT_INSTRUMENTATION=trace set?)", style="dim italic")
        else:
            body = _build_table(turns, pending)

        n = len(turns)
        title_parts = "[bold]robot-comic-monitor[/bold]" + title_suffix
        subtitle = f"[dim]{n} turn{'s' if n != 1 else ''} recorded[/dim]"
        return Panel(
            body,
            title=title_parts,
            subtitle=subtitle,
            border_style="green" if (watch_unit is None or _svc_active[0]) else "red",
        )

    stop_event = threading.Event()

    def _auto_refresh() -> None:
        while not stop_event.is_set():
            # Re-check service status every ~3 s.
            if watch_unit is not None:
                now = time.time()
                if now - _svc_last_check[0] >= 3.0:
                    _svc_active[0] = _service_active(watch_unit)
                    _svc_last_check[0] = now
            live.update(_render())
            stop_event.wait(0.15)  # ~6 fps for smooth spinner

    try:
        with Live(_render(), console=console, refresh_per_second=4, screen=True) as live:
            refresh_thread = threading.Thread(target=_auto_refresh, daemon=True)
            refresh_thread.start()

            for raw in lines:
                span = _parse_span(raw)
                if span is None:
                    continue
                turn = buffer.ingest(span)
                if turn is not None:
                    turns.append(turn)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
