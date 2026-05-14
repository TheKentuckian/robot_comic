"""Shared fixtures for integration smoke tests.

These fixtures bootstrap a handler in sim mode (no real robot, no real network)
and provide helpers for injecting synthetic transcripts and draining output queues.

All tests in this directory are marked ``integration`` and skipped by default.
Run them explicitly with::

    pytest -m integration tests/integration/ -v
"""

from __future__ import annotations
import os
import sys
import asyncio
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Path / environment bootstrap (mirrors tests/conftest.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[2].resolve()
SRC_PATH = PROJECT_ROOT / "src"
TESTS_PATH = PROJECT_ROOT / "tests"

for _p in (str(SRC_PATH), str(PROJECT_ROOT), str(TESTS_PATH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Prevent loading a developer's .env; keep tests hermetic.
os.environ.setdefault("REACHY_MINI_SKIP_DOTENV", "1")
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)
os.environ.setdefault("REACHY_MINI_HEAD_TRACKER", "off")
# Disable llama-server health check so the handler does not probe a real URL.
os.environ.setdefault("REACHY_MINI_LLAMA_HEALTH_CHECK", "0")
# Suppress TTS warmup calls during startup.
os.environ.setdefault("CHATTERBOX_WARMUP_ENABLED", "0")


# ---------------------------------------------------------------------------
# pytest marker auto-registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``integration`` marker so it appears in --markers output."""
    config.addinivalue_line(
        "markers",
        "integration: end-to-end smoke tests; skipped by default. Run with `pytest -m integration`.",
    )


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def make_tool_deps() -> Any:
    """Return a ToolDependencies instance with all hardware mocked."""
    from robot_comic.tools.core_tools import ToolDependencies

    robot = MagicMock()
    robot.media = MagicMock()
    robot.media.audio = None

    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True

    head_wobbler = MagicMock()

    return ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        head_wobbler=head_wobbler,
    )


def drain_queue(q: asyncio.Queue) -> list[Any]:  # type: ignore[type-arg]
    """Drain all items currently in *q* without blocking."""
    items: list[Any] = []
    while True:
        try:
            items.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


@pytest.fixture
def tool_deps() -> Any:
    """Pytest fixture wrapping :func:`make_tool_deps`."""
    return make_tool_deps()
