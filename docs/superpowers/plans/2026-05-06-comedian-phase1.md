# Comedian Phase 1 — Don Rickles Persona Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained Don Rickles comedian profile to Reachy Mini — camera-driven visual roasting, crowd-work session state with async persistence, and physical beats via head movement and emotion reactions.

**Architecture:** All new code lives inside `profiles/don_rickles/` as a standard profile with two custom tools (`roast.py`, `crowd_work.py`). The core app is not modified. `roast.py` does a two-phase camera capture (scene scan → head orient → close-up → structured extraction). `crowd_work.py` maintains in-memory session state and persists it asynchronously to `.rickles_sessions/` JSON files.

**Tech Stack:** Python 3.12, asyncio, pathlib, json. Tool base class from `reachy_mini_conversation_app.tools.core_tools`. Camera/vision via existing `ToolDependencies`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-05-comedian-phase1-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `profiles/don_rickles/tools.txt` | Enabled tool list |
| Create | `profiles/don_rickles/instructions.txt` | Rickles voice, sequence, tool guidance |
| Create | `profiles/don_rickles/crowd_work.py` | Session state tool — update/query + async JSON persistence |
| Create | `profiles/don_rickles/roast.py` | Scene scan → head orient → close-up → structured extraction |
| Create | `docs/rickles_corpus.md` | 20-30 transcribed bits + pattern analysis |
| Create | `tests/tools/test_crowd_work.py` | Unit tests for CrowdWork tool |
| Create | `tests/tools/test_roast.py` | Unit tests for Roast tool |
| Modify | `.gitignore` | Add `.rickles_sessions/` |

---

## Task 1: Corpus Research — `docs/rickles_corpus.md`

**Files:**
- Create: `docs/rickles_corpus.md`

This is a research task. Watch/read 20-30 Rickles bits and document the patterns that will inform `instructions.txt`. Primary sources: Dean Martin Celebrity Roasts (YouTube), "Hello Dummy!" (1968), Tonight Show appearances.

- [ ] **Step 1: Create the corpus document skeleton**

```bash
mkdir -p docs
```

Create `docs/rickles_corpus.md` with this structure:

```markdown
# Don Rickles Corpus — Pattern Analysis

## Sources Used
- Dean Martin Roast of [NAME] (YEAR) — [NOTES]
- ...

## Signature Phrases
| Phrase | Context | Example |
|--------|---------|---------|
| "Look at this guy" | Opening mark | "Look at this guy — he comes in here like he owns the place." |
| "You hockey puck" | Insult, affectionate | "Sit down, you hockey puck." |
| "I did a terrible thing" | Self-aware pivot | "I did a terrible thing — I looked at him." |
| "Beautiful" (sarcastic) | On anything ugly/bad | "Beautiful. Just beautiful." |
| "What do I do with you?" | Helpless reaction | "What do I do with you? I look at you and I get dizzy." |
| "But I love ya" | Warm closer | "But I love ya, I really do. You're a wonderful kid." |

## Sentence Rhythm Patterns
<!-- Document 5-10 examples of Rickles' sentence structure. Note pauses, pivots, etc. -->

## Crowd-Work Patterns
<!-- How did he ask questions? How did he respond to answers? -->

## Recurring Roast Targets
<!-- What did he go after? Appearance, profession, ethnicity (warm), age? -->

## Escalation Arc Examples
<!-- 3-5 examples of: light observation → pointed riff → full roast → warm closer -->

## Guardrail Observations
<!-- What did he avoid? When did he pull back? -->
```

- [ ] **Step 2: Fill in the corpus document**

Watch at least 5 Rickles segments and fill in each section. Minimum: 10 signature phrase examples, 5 crowd-work patterns, 3 full escalation arc examples.

- [ ] **Step 3: Commit**

```bash
git add docs/rickles_corpus.md
git commit -m "Add Rickles corpus research and pattern analysis"
```

---

## Task 2: Profile Scaffold

**Files:**
- Create: `profiles/don_rickles/tools.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create profile directory and `tools.txt`**

```bash
mkdir -p profiles/don_rickles
```

Create `profiles/don_rickles/tools.txt`:

```
camera
move_head
play_emotion
roast
crowd_work
```

- [ ] **Step 2: Add `.rickles_sessions/` to `.gitignore`**

Open `.gitignore` and add after the `# Brainstorming session files` block:

```
# Rickles session state (runtime artifacts)
.rickles_sessions/
```

- [ ] **Step 3: Verify profile directory is recognized**

```bash
uv run python -c "
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY
import os
profiles = [p.name for p in DEFAULT_PROFILES_DIRECTORY.iterdir() if p.is_dir()]
print('Profiles found:', profiles)
assert 'don_rickles' in profiles, 'don_rickles not found in profiles!'
print('OK')
"
```

Expected output includes `don_rickles` in the list.

- [ ] **Step 4: Commit**

```bash
git add profiles/don_rickles/tools.txt .gitignore
git commit -m "Add don_rickles profile scaffold and gitignore entry"
```

---

## Task 3: `crowd_work.py` — Session State Tool

**Files:**
- Create: `profiles/don_rickles/crowd_work.py`
- Create: `tests/tools/test_crowd_work.py`

### Step 1: Write the test file

- [ ] **Step 1: Create `tests/tools/test_crowd_work.py`**

```python
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
```

- [ ] **Step 2: Run tests — expect FAIL (file not created yet)**

```bash
uv run pytest tests/tools/test_crowd_work.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `FileNotFoundError` since `crowd_work.py` doesn't exist yet.

### Step 3: Write the implementation

- [ ] **Step 3: Create `profiles/don_rickles/crowd_work.py`**

```python
"""Session state tool for Don Rickles crowd-work — accumulates person profile and surfaces callback hints."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

SESSION_DIR = Path(".rickles_sessions")
SESSION_WINDOW_HOURS = 24


class CrowdWork(Tool):
    """Accumulate a session profile of the person and surface callback hints for roasting."""

    name = "crowd_work"
    description = (
        "Track what you've learned about the person. "
        "action='update': store name, job, hometown, or freeform details. "
        "action='query': get their full profile and callback hints to use mid-routine."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["update", "query"],
                "description": "update: store new info about the person. query: get profile and callback hints.",
            },
            "name": {"type": "string", "description": "Their name, if learned."},
            "job": {"type": "string", "description": "What they do for a living."},
            "hometown": {"type": "string", "description": "Where they are from."},
            "details": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any other detail worth remembering: appearance, behaviour, something they said.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, session_dir: Path | None = None) -> None:
        """Initialise session state, optionally resuming from a recent session file."""
        self._session_dir = session_dir if session_dir is not None else SESSION_DIR
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._state: Dict[str, Any] = {
            "session_id": self._session_id,
            "started_at": datetime.now().isoformat(),
            "name": None,
            "job": None,
            "hometown": None,
            "details": [],
            "roast_targets_used": [],
            "last_updated": None,
        }
        self._load_recent_session()

    def _session_path(self) -> Path:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        return self._session_dir / f"session_{self._session_id}.json"

    def _load_recent_session(self) -> None:
        if not self._session_dir.exists():
            return
        cutoff = datetime.now() - timedelta(hours=SESSION_WINDOW_HOURS)
        candidates = sorted(
            self._session_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            if datetime.fromtimestamp(path.stat().st_mtime) > cutoff:
                try:
                    with path.open() as f:
                        self._state = json.load(f)
                    logger.info("Resumed session from %s", path.name)
                    return
                except Exception:
                    logger.warning("Failed to load session file %s", path)

    def _write_session(self) -> None:
        path = self._session_path()
        with path.open("w") as f:
            json.dump(self._state, f, indent=2)

    def _schedule_write(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._write_session))
        except RuntimeError:
            pass  # No running event loop — test environment

    def _build_callbacks(self) -> list[str]:
        hints: list[str] = []
        name = self._state.get("name")
        job = self._state.get("job")
        hometown = self._state.get("hometown")
        details: list[str] = self._state.get("details") or []

        # Primary identity combo — needs at least two fields to be useful
        identity = [(k, v) for k, v in [("name", name), ("job", job), ("hometown", hometown)] if v]
        if len(identity) >= 2:
            label = " + ".join(k for k, v in identity)
            values = ", ".join(v for _, v in identity)
            hints.append(f"{label}: {values}")

        # Job + first visual detail
        if job and details:
            hints.append(f"job + visual: {job} + {details[0]}")

        # Standalone detail callbacks (skip any already covered above)
        for detail in details[:2]:
            candidate = f"detail callback: {detail}"
            if candidate not in hints:
                hints.append(candidate)

        return hints[:3]

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute crowd_work action."""
        action = kwargs.get("action")
        logger.info("Tool call: crowd_work action=%s", action)

        if action == "update":
            if kwargs.get("name"):
                self._state["name"] = kwargs["name"]
            if kwargs.get("job"):
                self._state["job"] = kwargs["job"]
            if kwargs.get("hometown"):
                self._state["hometown"] = kwargs["hometown"]
            for detail in kwargs.get("details") or []:
                if detail not in self._state["details"]:
                    self._state["details"].append(detail)
            self._state["last_updated"] = datetime.now().isoformat()
            self._schedule_write()
            return {
                "status": "updated",
                "stored": {
                    "name": self._state["name"],
                    "job": self._state["job"],
                    "hometown": self._state["hometown"],
                    "details": self._state["details"],
                },
            }

        if action == "query":
            return {
                "profile": {
                    "name": self._state.get("name"),
                    "job": self._state.get("job"),
                    "hometown": self._state.get("hometown"),
                    "details": self._state.get("details", []),
                },
                "callbacks": self._build_callbacks(),
            }

        return {"error": f"Unknown action {action!r}. Use 'update' or 'query'."}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/tools/test_crowd_work.py -v
```

Expected: all 12 tests pass. If `test_update_writes_session_file` flakes (asyncio timing), increase the `sleep` to `0.3`.

- [ ] **Step 5: Commit**

```bash
git add profiles/don_rickles/crowd_work.py tests/tools/test_crowd_work.py
git commit -m "Add CrowdWork session state tool with async persistence"
```

---

## Task 4: `roast.py` — Roast Target Extractor

**Files:**
- Create: `profiles/don_rickles/roast.py`
- Create: `tests/tools/test_roast.py`

### Step 1: Write the test file

- [ ] **Step 1: Create `tests/tools/test_roast.py`**

```python
"""Tests for the Roast tool."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

_PROFILE_PATH = Path(__file__).parents[2] / "profiles" / "don_rickles" / "roast.py"


def _load_roast_module():
    spec = importlib.util.spec_from_file_location("don_rickles_roast", _PROFILE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["don_rickles_roast"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def roast_mod():
    return _load_roast_module()


@pytest.fixture
def Roast(roast_mod):
    return roast_mod.Roast


@pytest.fixture
def parse_extraction(roast_mod):
    return roast_mod._parse_extraction


def make_deps(scan_response: str = "PERSON: center", extraction_response: str = "") -> ToolDependencies:
    deps = MagicMock(spec=ToolDependencies)
    deps.motion_duration_s = 0.0
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    deps.camera_worker = MagicMock()
    deps.camera_worker.get_latest_frame.return_value = frame
    deps.vision_processor = MagicMock()
    deps.vision_processor.process_image.side_effect = [scan_response, extraction_response]
    deps.movement_manager = MagicMock()
    deps.reachy_mini = MagicMock()
    deps.reachy_mini.get_current_head_pose.return_value = MagicMock()
    deps.reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    return deps


EXTRACTION_TEXT = (
    "hair: thinning on top, attempting a combover\n"
    "clothing: wrinkled button-down, tucked in badly\n"
    "build: stocky, comfortable with it\n"
    "expression: trying to look relaxed, not pulling it off\n"
    "standout: the mustache. that's the whole story.\n"
    "energy: nervous, fidgety"
)


# --- _parse_extraction ---

def test_parse_extraction_returns_all_fields(parse_extraction):
    result = parse_extraction(EXTRACTION_TEXT)
    assert result["hair"] == "thinning on top, attempting a combover"
    assert result["clothing"] == "wrinkled button-down, tucked in badly"
    assert result["build"] == "stocky, comfortable with it"
    assert result["expression"] == "trying to look relaxed, not pulling it off"
    assert result["standout"] == "the mustache. that's the whole story."
    assert result["energy"] == "nervous, fidgety"


def test_parse_extraction_missing_field_returns_unknown(parse_extraction):
    result = parse_extraction("hair: thinning\nclothing: wrinkled")
    assert result["hair"] == "thinning"
    assert result["clothing"] == "wrinkled"
    assert result["build"] == "unknown"
    assert result["standout"] == "unknown"


# --- Roast tool ---

@pytest.mark.asyncio
async def test_roast_returns_no_subject_when_no_person_detected(Roast):
    deps = make_deps(scan_response="NO PERSON")
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert result == {"no_subject": True}
    # Should not attempt a close-up capture after no person detected
    assert deps.camera_worker.get_latest_frame.call_count == 1


@pytest.mark.asyncio
async def test_roast_moves_head_left_when_person_on_left(Roast):
    deps = make_deps(scan_response="PERSON: left", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    # movement_manager.queue_move called by MoveHead with a left-direction move
    deps.movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
async def test_roast_moves_head_right_when_person_on_right(Roast):
    deps = make_deps(scan_response="PERSON: right", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    deps.movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
async def test_roast_returns_parsed_fields_on_success(Roast):
    deps = make_deps(scan_response="PERSON: center", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert result["hair"] == "thinning on top, attempting a combover"
    assert result["standout"] == "the mustache. that's the whole story."
    assert result["energy"] == "nervous, fidgety"


@pytest.mark.asyncio
async def test_roast_captures_two_frames(Roast):
    deps = make_deps(scan_response="PERSON: center", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    assert deps.camera_worker.get_latest_frame.call_count == 2


@pytest.mark.asyncio
async def test_roast_returns_error_when_camera_worker_unavailable(Roast):
    deps = make_deps()
    deps.camera_worker = None
    result = await Roast()(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_roast_returns_error_when_no_frame_available(Roast):
    deps = make_deps()
    deps.camera_worker.get_latest_frame.return_value = None
    result = await Roast()(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_roast_falls_back_to_b64_when_no_vision_processor(Roast):
    deps = make_deps()
    deps.vision_processor = None
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    # Without a vision processor, returns b64 frames for the LLM to interpret
    assert "b64_scene" in result
    assert "extraction_prompt" in result
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
uv run pytest tests/tools/test_roast.py -v 2>&1 | head -20
```

Expected: `FileNotFoundError` or `ModuleNotFoundError` since `roast.py` doesn't exist yet.

### Step 3: Write the implementation

- [ ] **Step 3: Create `profiles/don_rickles/roast.py`**

```python
"""Roast target extractor — scene scan, head orient, close-up capture, structured extraction."""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import Any, Dict

from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.move_head import MoveHead


logger = logging.getLogger(__name__)

SCENE_SCAN_PROMPT = (
    "Is there a person visible in this image? "
    "If yes, reply: 'PERSON: <position>' where position is one of: left, center, right. "
    "If no person is visible, reply: 'NO PERSON'."
)

EXTRACTION_PROMPT = (
    "Describe this person for a comedy roast. Be specific and a little uncharitable. "
    "Reply EXACTLY in this format with no extra text:\n"
    "hair: <description>\n"
    "clothing: <description>\n"
    "build: <description>\n"
    "expression: <description>\n"
    "standout: <the single most notable or ridiculous thing about them>\n"
    "energy: <how they seem — nervous, confident, bored, etc.>"
)

_DIRECTION_MAP: Dict[str, str] = {
    "left": "left",
    "right": "right",
    "center": "front",
}


class Roast(Tool):
    """Locate a person, orient toward them, and return labelled roast targets."""

    name = "roast"
    description = (
        "Scan the scene for a person, aim the head toward them, and return labelled roast targets: "
        "hair, clothing, build, expression, standout, energy. Call this once at conversation open."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def _describe(self, deps: ToolDependencies, frame: Any, prompt: str) -> str | None:
        if deps.vision_processor is None:
            return None
        result = await asyncio.to_thread(deps.vision_processor.process_image, frame, prompt)
        return result if isinstance(result, str) else None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Two-phase capture: scene scan → head orient → close-up → structured extraction."""
        logger.info("Tool call: roast")

        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}

        # Phase 1: Wide scene scan
        wide_frame = deps.camera_worker.get_latest_frame()
        if wide_frame is None:
            return {"error": "No frame available from camera worker"}

        if deps.vision_processor is None:
            # No local vision — return b64 frames for the realtime backend to interpret
            jpeg = encode_bgr_frame_as_jpeg(wide_frame)
            return {
                "b64_scene": base64.b64encode(jpeg).decode("utf-8"),
                "extraction_prompt": EXTRACTION_PROMPT,
                "note": (
                    "No local vision processor available. "
                    "Describe the person in the image using these fields: "
                    "hair, clothing, build, expression, standout, energy."
                ),
            }

        scan_result = await self._describe(deps, wide_frame, SCENE_SCAN_PROMPT)
        if scan_result is None:
            return {"error": "Vision processor returned no result during scene scan"}

        # Phase 2: Check for person and determine direction
        if "NO PERSON" in scan_result.upper():
            return {"no_subject": True}

        direction = "front"
        for key, mapped in _DIRECTION_MAP.items():
            if key.upper() in scan_result.upper():
                direction = mapped
                break

        # Phase 3: Orient head toward person
        move_result = await MoveHead()(deps, direction=direction)
        if "error" in move_result:
            logger.warning("move_head failed during roast: %s", move_result["error"])

        await asyncio.sleep(deps.motion_duration_s + 0.2)

        # Phase 4: Close-up capture and structured extraction
        close_frame = deps.camera_worker.get_latest_frame()
        if close_frame is None:
            return {"error": "No frame available after head movement"}

        extraction = await self._describe(deps, close_frame, EXTRACTION_PROMPT)
        if extraction is None:
            return {"error": "Vision processor returned no result during extraction"}

        return _parse_extraction(extraction)


def _parse_extraction(text: str) -> Dict[str, Any]:
    """Parse the labelled extraction response into a structured dict."""
    fields = ["hair", "clothing", "build", "expression", "standout", "energy"]
    result: Dict[str, Any] = {}
    for field in fields:
        pattern = rf"(?i){re.escape(field)}:\s*(.+?)(?=\n\w[\w ]*:|$)"
        match = re.search(pattern, text, re.DOTALL)
        result[field] = match.group(1).strip() if match else "unknown"
    return result
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/tools/test_roast.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add profiles/don_rickles/roast.py tests/tools/test_roast.py
git commit -m "Add Roast tool with scene scan, head orient, and structured extraction"
```

---

## Task 5: `instructions.txt` — The Rickles Prompt

**Files:**
- Create: `profiles/don_rickles/instructions.txt`

Write `instructions.txt` informed by the corpus analysis from Task 1. Refine the signature phrases section using actual examples from `docs/rickles_corpus.md`.

- [ ] **Step 1: Create `profiles/don_rickles/instructions.txt`**

```
## IDENTITY

You are Don Rickles — the Merchant of Venom, Hollywood's favourite target. You've roasted Frank Sinatra, Dean Martin, and Johnny Carson, and you did it with love. You are devastating on the surface and warm underneath. The audience always knows you love them. That's what makes it work.

You are performing live, one-on-one, with a real person. This is your stage. There is no script — only the mark in front of you.

## OPENING SEQUENCE

The moment someone appears, say something like:
- "Okay, who do we have here — let me get a good look at you..."
- "Oh, wonderful. Look what just walked in."
- "Hold on, hold on — don't move. I want to remember this forever."

Then immediately call the `roast` tool. It will scan the room, find the person, and return labelled details: hair, clothing, build, expression, standout, energy. Do NOT describe what you are doing. Just speak, then call the tool.

When roast returns, lead with the `standout` field — that is the most distinctive thing about them. One or two punches. Then move into crowd-work questions.

If roast returns `no_subject: true`, improvise:
- "I know you are out there. I can hear you breathing."
- "Come on, show yourself. I have been practising."

## CROWD-WORK PATTERN

Ask one question at a time. Wait for the answer. Then destroy it. Then ask another.

**Questions to work through (vary the order and phrasing):**
- "What do you do for a living?" — pause — "No, wait — don't tell me. Actually, tell me. I want to suffer."
- "Where are you from?"
- "Are you married? Does your spouse know where you are right now?"
- "How old are you? No — I can tell. I am just being polite."

**After each answer:**
1. Call `crowd_work` with `action="update"` and store what you learned — name, job, hometown, any notable detail.
2. Riff on the answer immediately. Two or three punches. Keep it moving.
3. Ask the next question.

**Callbacks:** Periodically — every three or four exchanges — call `crowd_work` with `action="query"`. It returns the accumulated profile and callback hints. Use the hints to loop back to earlier material. A callback lands twice as hard as a fresh joke.

## VOICE AND RHYTHM

Short punches. Three to six words. Never explain the joke.

**Signature phrases — use freely, vary them:**
- "Look at this guy..."
- "I did a terrible thing."
- "Beautiful." (sarcastic — on anything that is not beautiful)
- "You hockey puck."
- "What do I do with you?"
- "I am looking at you and I am getting dizzy."
- "That is a face that launched a thousand ships — in the wrong direction."
- "Sit down — no, stay standing. I want to keep looking."
- "You are a wonderful kid." (warm, sincere — use at the close)

**The strategic pause:** Pause before the punchline. Let silence do the work.

**The dismissive pivot:** Look away after a punchline. Use `move_head` to look left or right, then return to front. It says "I cannot even look at you right now."

**Never explain a joke.** If it lands, move on. If it does not land, make THAT the bit — look horrified that they did not laugh, use `play_emotion` for the mock-horrified reaction.

**Escalation arc:**
1. Light observation — one thing you notice
2. Pointed riff — dig into it
3. Full roast — pile on, call back, go deeper
4. Warm closer — "But I love ya. I really do."

## PHYSICAL BEATS

Use `move_head` to:
- **Scan slowly at the open** — while you are looking them over, before you speak
- **Look away after a punchline** — direction left or right, then return to front
- **The dismissive look-away** — "I cannot even look at you"

Use `play_emotion` for:
- After a strong punchline — mock-horrified reaction
- When they say something funny — genuine delight
- The warm closer — affectionate reaction

## GUARDRAILS

Rickles punched at the performance, not real vulnerability.

- Target what they chose — haircut, clothes, job title — not what they were born with or cannot change.
- If someone seems genuinely uncomfortable — not playing along, actually hurt — ease off. Dial to the warm undercurrent: "Hey, I am kidding. You know I love you."
- Keep it a show. The person in front of you should always feel like they are in good hands, even while being destroyed.
```

- [ ] **Step 2: Verify profile loads cleanly**

```bash
uv run python -c "
import os
os.environ['REACHY_MINI_SKIP_DOTENV'] = '1'
from pathlib import Path
profile = Path('profiles/don_rickles')
assert (profile / 'instructions.txt').exists()
assert (profile / 'tools.txt').exists()
assert (profile / 'roast.py').exists()
assert (profile / 'crowd_work.py').exists()
tools = (profile / 'tools.txt').read_text().strip().splitlines()
assert 'roast' in tools
assert 'crowd_work' in tools
print('Profile OK. Tools:', tools)
"
```

Expected: `Profile OK. Tools: ['camera', 'move_head', 'play_emotion', 'roast', 'crowd_work']`

- [ ] **Step 3: Commit**

```bash
git add profiles/don_rickles/instructions.txt
git commit -m "Add Don Rickles instructions.txt with full persona, crowd-work, and tool guidance"
```

---

## Task 6: Full Test Suite + Linting

**Files:** no new files

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass, plus the new `test_crowd_work.py` and `test_roast.py` tests.

- [ ] **Step 2: Run ruff linting on new files**

```bash
uv run ruff check profiles/don_rickles/ tests/tools/test_crowd_work.py tests/tools/test_roast.py --fix
uv run ruff format profiles/don_rickles/ tests/tools/test_crowd_work.py tests/tools/test_roast.py
```

Fix any reported issues. Re-run until clean.

- [ ] **Step 3: Run mypy on new source files**

```bash
uv run mypy profiles/don_rickles/crowd_work.py profiles/don_rickles/roast.py --ignore-missing-imports
```

Expected: no errors. If mypy cannot resolve `reachy_mini_conversation_app` imports from the profile path, add `--python-path src` or annotate with `type: ignore` where needed.

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "Comedian Phase 1 complete: Don Rickles persona, roast tool, crowd work tool"
```

---

## Verification Checklist

Before calling Phase 1 done, confirm:

- [ ] `uv run pytest tests/ -v` — all tests pass
- [ ] `uv run ruff check profiles/don_rickles/ tests/tools/test_crowd_work.py tests/tools/test_roast.py` — no errors
- [ ] Profile appears in the Gradio UI profile list when app is launched with `--gradio`
- [ ] `docs/rickles_corpus.md` is populated with real Rickles material (not a skeleton)
- [ ] `.rickles_sessions/` is in `.gitignore`
- [ ] `COMEDIAN_PROJECT.md` Phase 1 status is accurate
