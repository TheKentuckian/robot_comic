"""Persistent face-embedding database for repeat-visitor recognition.

Stores face embeddings + visitor names in ``~/.robot-comic/face-db.json``
and matches new embeddings via cosine similarity.

Enable with the ``REACHY_MINI_FACE_RECOGNITION_ENABLED`` env var (default
``False``).  Until a real :class:`FaceEmbedder` is wired in, this module is
a no-op from the application's perspective.

JSON schema (one entry per stored visitor)::

    [
        {
            "name": "Tony",
            "embedding": [0.12, -0.34, ...],   // list[float], unit-length recommended
            "first_seen": "2026-05-14T12:00:00+00:00",
            "last_seen": "2026-05-14T12:00:00+00:00",
            "session_count": 1
        },
        ...
    ]
"""

from __future__ import annotations
import os
import json
import math
import logging
import tempfile
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".robot-comic"
_DEFAULT_DB_FILENAME = "face-db.json"
_DEFAULT_MATCH_THRESHOLD = 0.85


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Return the cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector has zero norm to avoid division by zero.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def default_db_path() -> Path:
    """Return the default face-DB file path, creating the parent dir if needed."""
    try:
        _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create face-db directory %s: %s", _DEFAULT_DB_DIR, exc)
    return _DEFAULT_DB_DIR / _DEFAULT_DB_FILENAME


class FaceDatabase:
    """JSON-on-disk store for face embeddings and visitor metadata.

    Args:
        path: Path to the JSON file used for persistence.  Defaults to
            ``~/.robot-comic/face-db.json`` when not provided.
        threshold: Cosine-similarity threshold above which a match is
            reported.  Controlled at construction time so callers can tune
            it without touching env vars.

    All mutation methods (:meth:`add`, :meth:`update_last_seen`) persist
    changes atomically via a tmp-file rename so an interrupted write never
    corrupts the stored database.

    """

    def __init__(
        self,
        path: Path | None = None,
        threshold: float = _DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        """Initialise the database, loading any existing entries from *path*."""
        self._path: Path = path if path is not None else default_db_path()
        self._threshold = threshold
        self._entries: list[dict[str, Any]] = self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _load(self) -> list[dict[str, Any]]:
        """Load entries from disk.  Returns ``[]`` if the file is missing or unreadable."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning(
                "face-db file %s has unexpected format (not a list); starting fresh",
                self._path,
            )
        except Exception as exc:
            logger.warning("Could not read face-db from %s: %s", self._path, exc)
        return []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        """Atomically write *entries* to disk (tmp file + rename)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".face-db-",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(entries, fh, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Could not save face-db to %s: %s", self._path, exc)

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #

    def add(self, name: str, embedding: NDArray[np.float32]) -> None:
        """Append a new visitor entry and persist.

        Args:
            name: Human-readable visitor name (e.g. ``"Tony"``).
            embedding: 1-D float array produced by a :class:`FaceEmbedder`.

        """
        name = name.strip()
        if not name:
            logger.warning("face_db.add: empty name, skipping")
            return

        now = datetime.now(timezone.utc).isoformat()
        entry: dict[str, Any] = {
            "name": name,
            "embedding": embedding.tolist(),
            "first_seen": now,
            "last_seen": now,
            "session_count": 1,
        }
        self._entries.append(entry)
        self._save(self._entries)
        logger.debug("face_db.add: stored entry for %r (%d entries total)", name, len(self._entries))

    def update_last_seen(self, name: str) -> bool:
        """Bump ``last_seen`` timestamp and ``session_count`` for *name*.

        Finds the first entry whose ``name`` field matches (case-sensitive)
        and updates it in-place, then persists.

        Args:
            name: Visitor name to update (must match an existing entry exactly).

        Returns:
            ``True`` when an entry was found and updated, ``False`` otherwise.

        """
        name = name.strip()
        for entry in self._entries:
            if entry.get("name") == name:
                entry["last_seen"] = datetime.now(timezone.utc).isoformat()
                entry["session_count"] = int(entry.get("session_count", 0)) + 1
                self._save(self._entries)
                logger.debug(
                    "face_db.update_last_seen: %r now at session_count=%d",
                    name,
                    entry["session_count"],
                )
                return True
        logger.debug("face_db.update_last_seen: no entry found for %r", name)
        return False

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def match(
        self,
        embedding: NDArray[np.float32],
        threshold: float | None = None,
    ) -> dict[str, Any] | None:
        """Find the closest stored entry whose similarity exceeds *threshold*.

        Performs a linear scan over all stored embeddings and returns the
        entry with the highest cosine similarity, provided it is strictly
        above *threshold*.

        Args:
            embedding: 1-D float query embedding from a :class:`FaceEmbedder`.
            threshold: Override the instance-level threshold for this call.
                When ``None``, the value passed to ``__init__`` is used.

        Returns:
            The matching entry dict (same structure as stored in the JSON
            file) or ``None`` when the database is empty or no entry exceeds
            the threshold.

        """
        effective_threshold = threshold if threshold is not None else self._threshold

        best_entry: dict[str, Any] | None = None
        best_sim: float = -math.inf

        query = np.asarray(embedding, dtype=np.float32)

        for entry in self._entries:
            raw = entry.get("embedding")
            if not isinstance(raw, list) or not raw:
                continue
            stored = np.asarray(raw, dtype=np.float32)
            if stored.shape != query.shape:
                logger.debug(
                    "face_db.match: skipping entry %r — embedding shape mismatch (%s vs %s)",
                    entry.get("name"),
                    stored.shape,
                    query.shape,
                )
                continue
            sim = _cosine_similarity(query, stored)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim > effective_threshold:
            logger.debug(
                "face_db.match: matched %r with similarity=%.4f (threshold=%.4f)",
                best_entry.get("name"),
                best_sim,
                effective_threshold,
            )
            return dict(best_entry)

        if best_entry is not None:
            logger.debug(
                "face_db.match: best candidate %r at similarity=%.4f below threshold=%.4f — no match",
                best_entry.get("name"),
                best_sim,
                effective_threshold,
            )
        else:
            logger.debug("face_db.match: database is empty")
        return None

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the number of stored visitor entries."""
        return len(self._entries)
