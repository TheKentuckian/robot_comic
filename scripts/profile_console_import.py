#!/usr/bin/env python3
"""Profile what makes ``from robot_comic.console import LocalStream`` slow.

Run this on the robot (or anywhere the apps_venv is active):

    /venvs/apps_venv/bin/python scripts/profile_console_import.py

It re-executes the import under ``python -X importtime`` and surfaces:

  1. The top-level packages console.py imports, with cumulative time spent
     loading each (this is what you actually want to attribute the 8s to).
  2. The 30 slowest individual modules by self-time, across the whole graph
     (useful for spotting a single offender deep in a dependency).

Pass ``--target some.module`` to profile a different import.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

# Top-level imports console.py performs (kept in sync with src/robot_comic/console.py).
# We bucket cumulative load time by the first import of each of these names.
CONSOLE_TOP_LEVEL = [
    "os",
    "sys",
    "time",
    "asyncio",
    "logging",
    "typing",
    "pathlib",
    "fastrtc",
    "reachy_mini",
    "robot_comic.config",
    "robot_comic.pause_settings",
    "robot_comic.startup_settings",
    "robot_comic.audio.startup_config",
    "robot_comic.conversation_handler",
    "robot_comic.headless_personality_ui",
    "fastapi",
    "pydantic",
    "starlette",
]


LINE_RE = re.compile(
    r"^import time:\s+(?P<self>\d+)\s+\|\s+(?P<cum>\d+)\s+\|\s+(?P<indent>\s*)(?P<name>\S+)"
)


@dataclass
class Entry:
    self_us: int
    cum_us: int
    depth: int
    name: str


def parse_importtime(stderr: str) -> list[Entry]:
    entries: list[Entry] = []
    for line in stderr.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        entries.append(
            Entry(
                self_us=int(m.group("self")),
                cum_us=int(m.group("cum")),
                depth=len(m.group("indent")),
                name=m.group("name"),
            )
        )
    return entries


def fmt_ms(us: int) -> str:
    return f"{us / 1000:8.1f} ms"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target",
        default="from robot_comic.console import LocalStream",
        help="Python statement to execute under -X importtime",
    )
    p.add_argument("--top", type=int, default=30, help="How many slow modules to show")
    args = p.parse_args()

    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", args.target],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(f"\nTarget import failed (exit {proc.returncode}).\n")
        return proc.returncode

    entries = parse_importtime(proc.stderr)
    if not entries:
        sys.stderr.write("No importtime output parsed. Stderr was:\n")
        sys.stderr.write(proc.stderr)
        return 1

    # importtime emits one line per module when it finishes loading, ordered
    # by completion. The first occurrence of a given top-level name is its
    # full load (cumulative includes children).
    first_seen: dict[str, Entry] = {}
    for e in entries:
        if e.name not in first_seen:
            first_seen[e.name] = e

    print(f"Target: {args.target}")
    print(f"Total modules loaded: {len(entries)}")
    total_self = sum(e.self_us for e in entries)
    print(f"Sum of self times:    {fmt_ms(total_self)}\n")

    print("Cumulative cost of console.py's top-level imports")
    print("-" * 60)
    rows = []
    for name in CONSOLE_TOP_LEVEL:
        e = first_seen.get(name)
        if e is None:
            rows.append((0, name, "(not loaded)"))
        else:
            rows.append((e.cum_us, name, fmt_ms(e.cum_us)))
    rows.sort(key=lambda r: r[0], reverse=True)
    for cum, name, label in rows:
        print(f"  {label}  {name}")

    print()
    print(f"Top {args.top} slowest modules by SELF time (anywhere in the graph)")
    print("-" * 60)
    slowest = sorted(entries, key=lambda e: e.self_us, reverse=True)[: args.top]
    for e in slowest:
        print(f"  self={fmt_ms(e.self_us)}  cum={fmt_ms(e.cum_us)}  {e.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
