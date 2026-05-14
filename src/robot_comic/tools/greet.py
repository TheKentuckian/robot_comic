"""Greet tool — face presence scan, head sweep, and returning-visitor identity matching."""

from __future__ import annotations
import os
import json
import time
import asyncio
import difflib
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from robot_comic import config as _config
from robot_comic.tools.move_head import MoveHead
from robot_comic.tools.core_tools import Tool, ToolDependencies
from robot_comic.tools.crowd_work import resolve_session_dir
from robot_comic.tools.name_validation import validate_name_or_warn


logger = logging.getLogger(__name__)

SESSION_WINDOW_DAYS = 30
MATCH_THRESHOLD = 0.75
SWEEP_POSITIONS = ["left", "up", "right", "front"]

# How long (seconds) to wait for the face tracker to latch before giving up and
# sweeping.  Overrideable via REACHY_MINI_GREET_SCAN_WAIT_S.
_DEFAULT_SCAN_WAIT_S = 1.5
_SCAN_POLL_INTERVAL_S = 0.1

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
        "action='identify': attempt to identify the visitor. "
        "When face recognition is enabled, the camera frame is checked first — on a face match, returns "
        "returning=true with the recalled name and profile. "
        "If the face is not recognised, returns returning=false with a pending_embedding flag so the robot "
        "knows to ask the visitor's name and pass it to crowd_work update. "
        "When face recognition is disabled or unavailable, falls back to fuzzy name matching against stored "
        "sessions from the last 30 days: on match returns returning=true with profile and callback hints; "
        "on no match returns returning=false and name_received=<name>. "
        "The 'name' parameter is required when face recognition is disabled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "identify"],
                "description": (
                    "scan: detect face presence and sweep if needed. "
                    "identify: identify the visitor by face (if enabled) or spoken name."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "The name spoken by the person. "
                    "Optional when face recognition is enabled (face is checked first). "
                    "Required for action='identify' when face recognition is disabled."
                ),
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

        # Poll for up to REACHY_MINI_GREET_SCAN_WAIT_S seconds before sweeping.
        # On fresh startup the camera / tracker needs a moment to latch onto a
        # face even when one is already in frame; a single sample (or 3 rapid
        # retries) is not enough to bridge that initialisation gap.
        scan_wait_s = float(os.environ.get("REACHY_MINI_GREET_SCAN_WAIT_S", _DEFAULT_SCAN_WAIT_S))
        deadline = time.monotonic() + scan_wait_s
        while True:
            frame = deps.camera_worker.get_latest_frame()
            if frame is not None and _detect_face(frame):
                return {"face_detected": True}
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(_SCAN_POLL_INTERVAL_S)

        # Kill-switch: when this is set, skip the head sweep entirely and just
        # report no_subject. Used to keep the chassis safe while the discrete
        # 4-step sweep is observed to swing the head hard enough into the
        # cowling on at least one unit (tracked in #264). Default off so this
        # is opt-in via env until the proper velocity-clamped path lands.
        if os.environ.get("REACHY_MINI_GREET_SWEEP_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.info("greet: sweep disabled by REACHY_MINI_GREET_SWEEP_DISABLED; returning no_subject")
            return {"no_subject": True}

        for direction in SWEEP_POSITIONS:
            move_result = await MoveHead()(deps, direction=direction)
            if "error" in move_result:
                logger.warning("move_head failed during sweep (%s): %s", direction, move_result["error"])
            await asyncio.sleep(deps.motion_duration_s + 0.2)
            sweep_frame = deps.camera_worker.get_latest_frame()
            if sweep_frame is not None and _detect_face(sweep_frame):
                return {"face_detected": True}

        return {"no_subject": True}

    async def _identify_by_face(self, deps: ToolDependencies) -> Dict[str, Any] | None:
        """Attempt face-embedding based identification.

        Returns a result dict on success (match or no-match-with-embedding), or
        ``None`` when face recognition is disabled / unavailable / no frame.
        """
        if not _config.config.FACE_RECOGNITION_ENABLED:
            return None
        if deps.face_embedder is None or deps.face_db is None:
            return None

        frame = deps.camera_worker.get_latest_frame() if deps.camera_worker else None
        if frame is None:
            logger.debug("greet._identify_by_face: no frame available, skipping face path")
            return None

        embedding: "np.ndarray[Any, np.dtype[Any]] | None" = deps.face_embedder.embed(frame)
        if embedding is None:
            logger.debug("greet._identify_by_face: no face detected in frame")
            return {"returning": False, "name": None, "pending_embedding": True}

        match = deps.face_db.match(embedding)
        if match is not None:
            matched_name: str = match.get("name", "")
            logger.info("greet._identify_by_face: recognised %r", matched_name)
            return {
                "returning": True,
                "name": matched_name,
                "last_seen": match.get("last_seen"),
                "session_count": match.get("session_count"),
                "face_match": True,
                "load_instruction": "Call crowd_work action=update with name before proceeding.",
            }

        # Face detected but not in DB — return embedding as list for crowd_work persistence.
        logger.debug("greet._identify_by_face: face detected but no DB match")
        return {
            "returning": False,
            "name": None,
            "pending_embedding": embedding.tolist(),
        }

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
            # Face-recognition path: try to identify the visitor from their face
            # before falling back to spoken-name fuzzy matching.
            face_result = await self._identify_by_face(deps)
            if face_result is not None:
                return face_result

            # Name-based fallback (also the only path when face recognition is off).
            name = (kwargs.get("name") or "").strip()
            if not name:
                return {
                    "error": (
                        "name parameter is required for identify. "
                        "The person has not spoken their name yet. "
                        "Greet them and ask for their name first — do NOT invent or guess a name."
                    )
                }
            # Hallucination guard (#287): the LLM occasionally invents a name
            # the user never spoke.  Reject any name that does not appear
            # (word-boundary, case-insensitive) in the recent user transcripts
            # and tell the LLM to ask for the name instead of persisting a
            # fabricated one.
            if not validate_name_or_warn(
                name,
                deps.recent_user_transcripts,
                tool_name="greet.identify",
            ):
                return {
                    "returning": False,
                    "name_received": None,
                    "needs_name": True,
                    "note": (
                        "The provided name was not heard in recent user speech. "
                        "Ask the visitor for their name before calling identify again."
                    ),
                }
            return await self._identify(deps, name)

        return {"error": f"Unknown action {action!r}. Use 'scan' or 'identify'."}
