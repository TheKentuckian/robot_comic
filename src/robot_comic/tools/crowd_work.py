"""Session state tool for crowd-work — accumulates person profile and surfaces callback hints."""

from __future__ import annotations
import json
import asyncio
import logging
from typing import Any, Dict
from pathlib import Path
from datetime import datetime

import numpy as np

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

SESSION_DIRNAME = ".comedy_sessions"
SESSION_WINDOW_HOURS = 4


def resolve_session_dir() -> Path:
    """Return the comedy-sessions directory, always rooted under ~/.robot_comic."""
    session_dir = Path.home() / ".robot_comic" / SESSION_DIRNAME
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


class CrowdWork(Tool):
    """Accumulate a session profile of the person and surface callback hints for the routine."""

    name = "crowd_work"
    description = (
        "Track what you've learned about the person. "
        "action='update': store name, job, hometown, freeform details, or roast targets already used as punchlines. "
        "action='query': get their full profile, callback hints, and which roast targets have already been used. "
        "action='clear': reset all in-memory state and start a fresh session (use when the user wants a fresh start)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["update", "query", "clear"],
                "description": (
                    "update: store new info about the person. "
                    "query: get profile and callback hints. "
                    "clear: reset all in-memory session state (name, job, hometown, details, "
                    "roast_targets_used) and start a fresh session ID — use when the user asks "
                    "to start over, pretend you don't know them, or have a fresh start."
                ),
            },
            "name": {"type": "string", "description": "Their name, if learned."},
            "job": {"type": "string", "description": "What they do for a living."},
            "hometown": {"type": "string", "description": "Where they are from."},
            "details": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any other detail worth remembering: appearance, behaviour, something they said.",
            },
            "roast_targets_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Roast fields from the 'roast' tool that have already been used as punchlines "
                    "(e.g. 'hair', 'clothing', 'build', 'expression', 'standout', 'energy'). "
                    "Pass after delivering a punchline so these are not repeated."
                ),
            },
            "pending_embedding": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "128-D face embedding list returned by greet action='identify' when the visitor "
                    "was not recognised (pending_embedding field in that response). "
                    "Pass it here together with the visitor's name so the face is stored for future "
                    "sessions. Omit when no pending_embedding was returned by greet."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self, session_dir: Path | None = None) -> None:
        """Initialise state. Resume the most recent same-day session lazily on first call.

        Resume is deferred to first call since singleton tools are constructed
        before the session dir is needed. When ``session_dir`` is passed
        explicitly (tests), eager-resume is preserved.
        """
        self._explicit_session_dir: Path | None = session_dir
        self._session_dir: Path | None = session_dir
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
        self._resumed = False
        if session_dir is not None:
            self._load_recent_session()
            self._resumed = True

    def _ensure_session_dir(self, deps: ToolDependencies) -> None:
        """Resolve the session directory from deps on first use and load any same-day session."""
        if self._explicit_session_dir is not None:
            return
        if self._session_dir is None:
            self._session_dir = resolve_session_dir()
        if not self._resumed:
            self._load_recent_session()
            self._resumed = True

    def _session_path(self) -> Path:
        assert self._session_dir is not None
        return self._session_dir / f"session_{self._session_id}.json"

    def _load_recent_session(self) -> None:
        assert self._session_dir is not None
        if not self._session_dir.exists():
            return
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
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
                    self._session_id = self._state.get("session_id", self._session_id)
                    logger.info("Resumed session from %s", path.name)
                    return
                except Exception:
                    logger.warning("Failed to load session file %s", path)

    def _write_session(self) -> None:
        assert self._session_dir is not None
        self._session_dir.mkdir(parents=True, exist_ok=True)
        path = self._session_path()
        with path.open("w") as f:
            json.dump(self._state, f, indent=2)

    def _schedule_write(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._write_session))
        except RuntimeError:
            logger.warning("No running event loop — session write skipped")

    def _build_callbacks(self) -> list[str]:
        hints: list[str] = []
        name = self._state.get("name")
        job = self._state.get("job")
        hometown = self._state.get("hometown")
        details: list[str] = self._state.get("details") or []

        identity = [(k, v) for k, v in [("name", name), ("job", job), ("hometown", hometown)] if v]
        if len(identity) >= 2:
            label = " + ".join(k for k, v in identity)
            values = ", ".join(v for _, v in identity)
            hints.append(f"{label}: {values}")

        if job and details:
            hints.append(f"job + visual: {job} + {details[0]}")

        # Standalone detail callbacks — skip details already referenced above
        already_in_hints = {details[0]} if (job and details) else set()
        for detail in details[:2]:
            if detail not in already_in_hints:
                hints.append(f"detail callback: {detail}")
                already_in_hints.add(detail)

        return hints[:3]

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch to update or query based on the required 'action' kwarg."""
        self._ensure_session_dir(deps)
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
            for target in kwargs.get("roast_targets_used") or []:
                if target not in self._state["roast_targets_used"]:
                    self._state["roast_targets_used"].append(target)
            self._state["last_updated"] = datetime.now().isoformat()
            self._schedule_write()

            # Persist face embedding when the LLM passes a pending_embedding from
            # greet(action='identify').  Best-effort: failure is logged but not raised.
            pending_raw = kwargs.get("pending_embedding")
            stored_name: str | None = self._state.get("name")
            if pending_raw is not None and stored_name and deps.face_db is not None:
                try:
                    embedding = np.asarray(pending_raw, dtype=np.float64)
                    deps.face_db.add(stored_name, embedding)
                    logger.info(
                        "crowd_work: persisted face embedding for %r (%d-D)",
                        stored_name,
                        len(embedding),
                    )
                except Exception as exc:
                    logger.warning("crowd_work: failed to persist face embedding: %s", exc)

            return {
                "status": "updated",
                "stored": {
                    "name": self._state["name"],
                    "job": self._state["job"],
                    "hometown": self._state["hometown"],
                    "details": self._state["details"],
                    "roast_targets_used": self._state["roast_targets_used"],
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
                "roast_targets_used": self._state.get("roast_targets_used", []),
            }

        if action == "clear":
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._state = {
                "session_id": self._session_id,
                "started_at": datetime.now().isoformat(),
                "name": None,
                "job": None,
                "hometown": None,
                "details": [],
                "roast_targets_used": [],
                "last_updated": None,
            }
            logger.info("Session cleared — new session_id=%s", self._session_id)
            return {"action": "clear", "ok": True}

        return {"error": f"Unknown action {action!r}. Use 'update', 'query', or 'clear'."}
