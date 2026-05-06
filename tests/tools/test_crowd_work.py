"""Tests for the CrowdWork session state tool."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Load CrowdWork from its profile path (not on sys.path)
_PROFILE_PATH = Path(__file__).parents[2] / "profiles" / "don_rickles" / "crowd_work.py"


def _load_crowd_work():
    spec = importlib.util.spec_from_file_location("don_rickles_crowd_work", _PROFILE_PATH)
    assert spec and spec.loader, f"Cannot load module from {_PROFILE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["don_rickles_crowd_work"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CrowdWork


def make_deps() -> MagicMock:
    return MagicMock()


@pytest.fixture
def CrowdWork():
    return _load_crowd_work()


@pytest.fixture
def crowd_work(tmp_path, CrowdWork):
    session_dir = tmp_path / ".rickles_sessions"
    return CrowdWork(session_dir=session_dir)


# --- update action ---

@pytest.mark.asyncio
async def test_update_stores_name_job_hometown(crowd_work):
    result = await crowd_work(make_deps(), action="update", name="Tony", job="engineer", hometown="Pittsburgh")
    assert result["status"] == "updated"
    assert crowd_work._state["name"] == "Tony"
    assert crowd_work._state["job"] == "engineer"
    assert crowd_work._state["hometown"] == "Pittsburgh"


@pytest.mark.asyncio
async def test_update_appends_details_without_duplicates(crowd_work):
    await crowd_work(make_deps(), action="update", details=["nervous laugh"])
    await crowd_work(make_deps(), action="update", details=["nervous laugh", "wrinkled shirt"])
    assert crowd_work._state["details"] == ["nervous laugh", "wrinkled shirt"]


@pytest.mark.asyncio
async def test_update_partial_fields_does_not_overwrite_others(crowd_work):
    await crowd_work(make_deps(), action="update", name="Tony", job="engineer")
    await crowd_work(make_deps(), action="update", hometown="Pittsburgh")
    assert crowd_work._state["name"] == "Tony"
    assert crowd_work._state["job"] == "engineer"
    assert crowd_work._state["hometown"] == "Pittsburgh"


# --- query action ---

@pytest.mark.asyncio
async def test_query_returns_empty_callbacks_when_no_data(crowd_work):
    result = await crowd_work(make_deps(), action="query")
    assert result["profile"]["name"] is None
    assert result["profile"]["job"] is None
    assert result["profile"]["hometown"] is None
    assert result["callbacks"] == []


@pytest.mark.asyncio
async def test_query_returns_callbacks_with_two_or_more_identity_fields(crowd_work):
    await crowd_work(make_deps(), action="update", name="Tony", job="engineer")
    result = await crowd_work(make_deps(), action="query")
    assert len(result["callbacks"]) >= 1
    assert "Tony" in result["callbacks"][0]
    assert "engineer" in result["callbacks"][0]


@pytest.mark.asyncio
async def test_query_includes_detail_callbacks(crowd_work):
    await crowd_work(make_deps(), action="update", details=["nervous laugh", "wrinkled shirt"])
    result = await crowd_work(make_deps(), action="query")
    assert any("nervous laugh" in cb for cb in result["callbacks"])


@pytest.mark.asyncio
async def test_query_returns_at_most_three_callbacks(crowd_work):
    await crowd_work(
        make_deps(),
        action="update",
        name="Tony", job="engineer", hometown="Pittsburgh",
        details=["nervous laugh", "wrinkled shirt", "combover"],
    )
    result = await crowd_work(make_deps(), action="query")
    assert len(result["callbacks"]) <= 3


# --- unknown action ---

@pytest.mark.asyncio
async def test_unknown_action_returns_error(crowd_work):
    result = await crowd_work(make_deps(), action="explode")
    assert "error" in result


# --- persistence ---

@pytest.mark.asyncio
async def test_update_writes_session_file(crowd_work, tmp_path):
    await crowd_work(make_deps(), action="update", name="Tony")
    await asyncio.sleep(0.15)  # let fire-and-forget write complete
    session_dir = tmp_path / ".rickles_sessions"
    files = list(session_dir.glob("session_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["name"] == "Tony"
    assert "session_id" in data
    assert "started_at" in data


@pytest.mark.asyncio
async def test_load_recent_session_resumes_from_disk(tmp_path, CrowdWork):
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    session_file = session_dir / "session_20260506_120000.json"
    session_file.write_text(json.dumps({
        "session_id": "20260506_120000",
        "started_at": "2026-05-06T12:00:00",
        "name": "Tony",
        "job": "engineer",
        "hometown": "Pittsburgh",
        "details": ["nervous laugh"],
        "roast_targets_used": [],
        "last_updated": "2026-05-06T12:05:00",
    }))
    cw = CrowdWork(session_dir=session_dir)
    assert cw._state["name"] == "Tony"
    assert cw._state["job"] == "engineer"


@pytest.mark.asyncio
async def test_load_does_not_resume_old_session(tmp_path, CrowdWork):
    """Sessions outside SESSION_WINDOW_HOURS should not be loaded."""
    session_dir = tmp_path / ".rickles_sessions"
    session_dir.mkdir()
    session_file = session_dir / "session_19990101_120000.json"
    old_data = {
        "session_id": "19990101_120000",
        "started_at": "1999-01-01T12:00:00",
        "name": "OldPerson",
        "job": "dinosaur wrangler",
        "hometown": "Jurassic Park",
        "details": [],
        "roast_targets_used": [],
        "last_updated": "1999-01-01T12:05:00",
    }
    session_file.write_text(json.dumps(old_data))
    # Force old mtime
    import os, time
    old_time = 946728000  # 2000-01-01 approximately
    os.utime(session_file, (old_time, old_time))
    cw = CrowdWork(session_dir=session_dir)
    assert cw._state["name"] is None
