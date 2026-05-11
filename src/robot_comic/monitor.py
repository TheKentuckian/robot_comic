"""Live TUI monitor for robot_comic turn traces.

Tails the systemd journal (or a log file) for RCSPAN lines emitted by
CompactLineExporter and renders a rolling table of completed turns.

Usage:
    robot-comic-monitor                          # tail reachy-app-autostart unit
    robot-comic-monitor --unit my-service        # different systemd unit
    robot-comic-monitor --file /path/to/app.log  # tail a file instead
"""

import sys
import json
import time
import argparse
import subprocess
from typing import Iterator, Optional
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
    "stt":   (300,  600),
    "llm":   (800,  2000),
    "tts":   (500,  1500),
    "total": (2000, 4000),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TurnRecord:
    """Aggregates all spans for one conversational turn."""

    def __init__(self, trace_id: str, root: dict, children: list[dict]) -> None:
        """Build a TurnRecord from the root turn span and its buffered children."""
        self.trace_id = trace_id
        self.root = root
        self.children = children
        self.ts = datetime.fromtimestamp(root["ts"] / 1000)

    def _kids(self, name: str) -> list[dict]:
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
        return self.root["dur_ms"]

    @property
    def outcome(self) -> str:
        """Turn outcome attribute."""
        return self.root["attrs"].get("turn.outcome", "?")

    @property
    def mode(self) -> str:
        """Backend mode (openai, chatterbox, gemini, etc.)."""
        return self.root["attrs"].get("robot.mode", "?")

    @property
    def tool_count(self) -> int:
        """Number of tool calls made during this turn."""
        return len(self._kids("tool.execute"))


# ---------------------------------------------------------------------------
# Span buffer
# ---------------------------------------------------------------------------

class SpanBuffer:
    """Accumulates child spans until the root turn span closes the trace."""

    def __init__(self) -> None:
        """Initialise the pending-span store."""
        self._pending: dict[str, list[dict]] = defaultdict(list)

    def ingest(self, span: dict) -> Optional[TurnRecord]:
        """Return a completed TurnRecord when the root turn span arrives, else None."""
        trace_id = span["trace"]
        if span["name"] == "turn" and span["parent"] is None:
            children = self._pending.pop(trace_id, [])
            return TurnRecord(trace_id, span, children)
        self._pending[trace_id].append(span)
        return None


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


def _build_table(turns: list[TurnRecord]) -> Table:
    """Render the most recent turns as a Rich table (newest row first)."""
    t = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style="bold dim",
        row_styles=["", "dim"],
    )
    t.add_column("Time",  width=8,  no_wrap=True)
    t.add_column("Mode",  width=11, no_wrap=True)
    t.add_column("STT",   width=8,  justify="right")
    t.add_column("LLM",   width=8,  justify="right")
    t.add_column("TTS",   width=8,  justify="right")
    t.add_column("Total", width=8,  justify="right")
    t.add_column("Tools", width=5,  justify="right")
    t.add_column("",      width=1,  justify="center")

    for turn in reversed(turns[-_MAX_TURNS:]):
        ok_icon = Text("✓", style="green") if turn.outcome == "success" else Text("✗", style="red")
        tools_cell = Text(str(turn.tool_count), style="cyan") if turn.tool_count else Text("")
        t.add_row(
            turn.ts.strftime("%H:%M:%S"),
            turn.mode,
            _fmt(turn.stt_ms,   "stt"),
            _fmt(turn.llm_ms,   "llm"),
            _fmt(turn.tts_ms,   "tts"),
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


def _parse_span(line: str) -> Optional[dict]:
    """Extract a span dict from a line containing an RCSPAN JSON payload."""
    idx = line.find("RCSPAN ")
    if idx == -1:
        return None
    try:
        return json.loads(line[idx + 7:])
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
        "--unit", default=_JOURNALD_UNIT, metavar="UNIT",
        help=f"systemd unit to tail (default: {_JOURNALD_UNIT})",
    )
    source.add_argument(
        "--file", metavar="PATH",
        help="tail a log file instead of the systemd journal",
    )
    args = parser.parse_args()

    console = Console()
    buffer = SpanBuffer()
    turns: list[TurnRecord] = []

    lines: Iterator[str] = _iter_file(args.file) if args.file else _iter_journald(args.unit)
    source_label = f"file:{args.file}" if args.file else f"journald:{args.unit}"

    def _render() -> Panel:
        if not turns:
            body = Text("Waiting for turns… (is ROBOT_INSTRUMENTATION=trace set?)", style="dim italic")
        else:
            body = _build_table(turns)
        return Panel(
            body,
            title=f"[bold]robot-comic-monitor[/bold]  [dim]{source_label}[/dim]",
            subtitle=f"[dim]{len(turns)} turn{'s' if len(turns) != 1 else ''} recorded[/dim]",
            border_style="dim",
        )

    try:
        with Live(_render(), console=console, refresh_per_second=4, screen=False) as live:
            for raw in lines:
                span = _parse_span(raw)
                if span is None:
                    continue
                turn = buffer.ingest(span)
                if turn is not None:
                    turns.append(turn)
                live.update(_render())
    except KeyboardInterrupt:
        pass
