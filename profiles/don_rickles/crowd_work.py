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
            pass

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

        for detail in details[:2]:
            candidate = f"detail callback: {detail}"
            if candidate not in hints:
                hints.append(candidate)

        return hints[:3]

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
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
