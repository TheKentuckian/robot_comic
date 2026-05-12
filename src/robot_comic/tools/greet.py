"""Greet tool — face presence scan, head sweep, and returning-visitor identity matching."""

from __future__ import annotations
import json
import asyncio
import difflib
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta

from robot_comic.tools.move_head import MoveHead
from robot_comic.tools.core_tools import Tool, ToolDependencies
from robot_comic.tools.crowd_work import resolve_session_dir


logger = logging.getLogger(__name__)

SESSION_WINDOW_DAYS = 30
MATCH_THRESHOLD = 0.75
SWEEP_POSITIONS = ["left", "up", "right", "front"]

try:
    import mediapipe as mp

    _mp_face_detection = mp.solutions.face_detection
    MP_AVAILABLE = True
except ImportError:
    _mp_face_detection = None
    MP_AVAILABLE = False


def _detect_face(frame: Any) -> bool:
    """Return True if MediaPipe detects at least one face in the BGR frame."""
    if not MP_AVAILABLE or _mp_face_detection is None:
        return False
    rgb = frame[..., ::-1].copy()
    with _mp_face_detection.FaceDetection(min_detection_confidence=0.3) as detector:
        results = detector.process(rgb)
        return bool(results.detections)


def _build_callbacks(state: Dict[str, Any]) -> List[str]:
    """Build callback hints from session state — mirrors CrowdWork._build_callbacks."""
    hints: List[str] = []
    name = state.get("name")
    job = state.get("job")
    hometown = state.get("hometown")
    details: List[str] = state.get("details") or []

    identity = [(k, v) for k, v in [("name", name), ("job", job), ("hometown", hometown)] if v]
    if len(identity) >= 2:
        label = " + ".join(k for k, v in identity)
        values = ", ".join(v for _, v in identity)
        hints.append(f"{label}: {values}")

    if job and details:
        hints.append(f"job + visual: {job} + {details[0]}")

    already_in_hints = {details[0]} if (job and details) else set()
    for detail in details[:2]:
        if detail not in already_in_hints:
            hints.append(f"detail callback: {detail}")
            already_in_hints.add(detail)

    return hints[:3]


def _fuzzy_match(
    query: str,
    candidates: List[Tuple[str, Path]],
) -> Tuple[Optional[str], Optional[Path], float]:
    """Return (best_name, best_path, score) for the highest match >= MATCH_THRESHOLD, else (None, None, score).

    Tries both the full query and the first word so that "Tony Anzelmo" still
    matches a stored name of "Tony" (full-string ratio would be ~0.5; first-word
    ratio is 1.0).
    """
    best_name: Optional[str] = None
    best_path: Optional[Path] = None
    best_score = 0.0
    q_full = query.lower().strip()
    q_tokens = q_full.split()
    for name, path in candidates:
        n = name.lower().strip()
        n_first = n.split()[0] if n else n
        scores = [difflib.SequenceMatcher(None, q_full, n).ratio()]
        for tok in q_tokens:
            scores.append(difflib.SequenceMatcher(None, tok, n).ratio())
            scores.append(difflib.SequenceMatcher(None, tok, n_first).ratio())
        score = max(scores)
        if score > best_score:
            best_score, best_name, best_path = score, name, path
    if best_score >= MATCH_THRESHOLD:
        return best_name, best_path, best_score
    return None, None, best_score


def _load_candidates(session_dir: Path) -> List[Tuple[str, Path]]:
    """Return (name, path) pairs from session files modified within SESSION_WINDOW_DAYS."""
    if not session_dir.exists():
        return []
    cutoff = datetime.now() - timedelta(days=SESSION_WINDOW_DAYS)
    candidates: List[Tuple[str, Path]] = []
    for path in session_dir.glob("session_*.json"):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name")
            if name and isinstance(name, str):
                candidates.append((name, path))
        except Exception:
            logger.warning("Skipping unreadable session file %s", path)
    return candidates


class Greet(Tool):
    """Detect face presence and identify returning visitors by name."""

    name = "greet"
    description = (
        "Two actions. "
        "action='scan': detect whether a face is present; executes a slow head sweep if not found immediately. "
        "action='identify': fuzzy-match a spoken name against stored sessions from the last 30 days. "
        "On match returns returning=true with profile and callback hints. "
        "On no match returns returning=false and name_received=<name> — use name_received to address the new visitor by name without asking again."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "identify"],
                "description": (
                    "scan: detect face presence and sweep if needed. "
                    "identify: match spoken name against stored sessions."
                ),
            },
            "name": {
                "type": "string",
                "description": "The name spoken by the person. Required for action='identify'.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, session_dir: Optional[Path] = None) -> None:
        """Initialise with an optional session directory override (for testing).

        When no override is provided, the directory resolves to
        ~/.robot_comic/.comedy_sessions/ on first call.
        """
        self._explicit_session_dir: Optional[Path] = session_dir
        self._session_dir: Optional[Path] = session_dir

    def _ensure_session_dir(self, deps: ToolDependencies) -> None:
        if self._explicit_session_dir is not None:
            return
        if self._session_dir is None:
            self._session_dir = resolve_session_dir()

    async def _scan(self, deps: ToolDependencies) -> Dict[str, Any]:
        if deps.camera_worker is None:
            return {"error": "Camera not available"}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No frame available"}

        if not MP_AVAILABLE:
            return {"face_detected": True, "note": "MediaPipe unavailable, assuming face present"}

        # Try up to 3 times on the initial front frame before sweeping (camera may need to settle)
        for _ in range(3):
            frame = deps.camera_worker.get_latest_frame()
            if frame is not None and _detect_face(frame):
                return {"face_detected": True}
            await asyncio.sleep(0.3)

        for direction in SWEEP_POSITIONS:
            move_result = await MoveHead()(deps, direction=direction)
            if "error" in move_result:
                logger.warning("move_head failed during sweep (%s): %s", direction, move_result["error"])
            await asyncio.sleep(deps.motion_duration_s + 0.2)
            sweep_frame = deps.camera_worker.get_latest_frame()
            if sweep_frame is not None and _detect_face(sweep_frame):
                return {"face_detected": True}

        return {"no_subject": True}

    async def _identify(self, deps: ToolDependencies, name: str) -> Dict[str, Any]:
        assert self._session_dir is not None
        candidates = _load_candidates(self._session_dir)
        if not candidates:
            return {"returning": False, "name_received": name}

        matched_name, matched_path, _score = _fuzzy_match(name, candidates)
        if matched_name is None or matched_path is None:
            return {"returning": False, "name_received": name}

        try:
            state = json.loads(matched_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read matched session %s", matched_path)
            return {"returning": False, "name_received": name}

        last_seen = state.get("last_updated") or str(datetime.fromtimestamp(matched_path.stat().st_mtime).isoformat())
        profile = {
            "job": state.get("job"),
            "hometown": state.get("hometown"),
            "details": state.get("details", []),
        }
        callbacks = _build_callbacks(state)

        return {
            "returning": True,
            "name": matched_name,
            "last_seen": last_seen,
            "profile": profile,
            "callbacks": callbacks,
            "load_instruction": "Call crowd_work action=update with this profile before proceeding.",
        }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch to scan or identify."""
        self._ensure_session_dir(deps)
        action = kwargs.get("action")
        logger.info("Tool call: greet action=%s", action)

        if action == "scan":
            return await self._scan(deps)

        if action == "identify":
            name = (kwargs.get("name") or "").strip()
            if not name:
                return {
                    "error": (
                        "name parameter is required for identify. "
                        "The person has not spoken their name yet. "
                        "Greet them and ask for their name first — do NOT invent or guess a name."
                    )
                }
            return await self._identify(deps, name)

        return {"error": f"Unknown action {action!r}. Use 'scan' or 'identify'."}
