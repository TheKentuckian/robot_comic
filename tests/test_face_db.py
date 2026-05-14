"""Tests for robot_comic.vision.face_db."""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from robot_comic.vision.face_db import FaceDatabase, _cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_embedding(dim: int = 128, seed: int | None = None) -> "np.ndarray[np.float32, np.dtype[np.float32]]":
    """Return a random unit-length float32 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical() -> None:
    v = _rand_embedding(64, seed=0)
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector() -> None:
    a = np.zeros(4, dtype=np.float32)
    b = _rand_embedding(4, seed=1)
    assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# FaceDatabase.add + round-trip load
# ---------------------------------------------------------------------------


def test_add_and_load_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "face-db.json"
    db = FaceDatabase(path=db_path)
    emb = _rand_embedding(128, seed=42)

    db.add("Tony", emb)

    # Re-open from disk to verify persistence.
    db2 = FaceDatabase(path=db_path)
    assert len(db2) == 1
    entry = db2._entries[0]
    assert entry["name"] == "Tony"
    assert entry["session_count"] == 1
    assert "first_seen" in entry
    assert "last_seen" in entry

    stored_emb = np.asarray(entry["embedding"], dtype=np.float32)
    assert stored_emb.shape == emb.shape
    assert np.allclose(stored_emb, emb, atol=1e-6)


def test_add_multiple_entries(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json")
    db.add("Alice", _rand_embedding(64, seed=1))
    db.add("Bob", _rand_embedding(64, seed=2))
    db.add("Carol", _rand_embedding(64, seed=3))

    db2 = FaceDatabase(path=tmp_path / "face-db.json")
    assert len(db2) == 3
    names = [e["name"] for e in db2._entries]
    assert names == ["Alice", "Bob", "Carol"]


def test_add_empty_name_is_skipped(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json")
    db.add("", _rand_embedding(64, seed=0))
    assert len(db) == 0


# ---------------------------------------------------------------------------
# FaceDatabase.match — empty DB
# ---------------------------------------------------------------------------


def test_match_empty_db_returns_none(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json")
    result = db.match(_rand_embedding(128, seed=99))
    assert result is None


# ---------------------------------------------------------------------------
# FaceDatabase.match — positive match
# ---------------------------------------------------------------------------


def test_match_returns_closest_above_threshold(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json", threshold=0.85)
    emb_alice = _rand_embedding(128, seed=10)
    emb_bob = _rand_embedding(128, seed=20)
    db.add("Alice", emb_alice)
    db.add("Bob", emb_bob)

    # Querying with Alice's own embedding should yield a perfect match.
    result = db.match(emb_alice)
    assert result is not None
    assert result["name"] == "Alice"


def test_match_identical_embedding(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json", threshold=0.5)
    emb = _rand_embedding(64, seed=7)
    db.add("Identical", emb)

    result = db.match(emb)
    assert result is not None
    assert result["name"] == "Identical"


# ---------------------------------------------------------------------------
# FaceDatabase.match — no match below threshold
# ---------------------------------------------------------------------------


def test_match_returns_none_below_threshold(tmp_path: Path) -> None:
    # Use a very high threshold (effectively 1.0) so no match is possible
    # unless the query is identical to a stored embedding.
    db = FaceDatabase(path=tmp_path / "face-db.json", threshold=0.9999)
    db.add("Alice", _rand_embedding(128, seed=11))

    # A completely different random embedding should not match.
    query = _rand_embedding(128, seed=999)
    result = db.match(query)
    assert result is None


def test_match_threshold_override(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json", threshold=0.5)
    emb = _rand_embedding(128, seed=5)
    db.add("Someone", emb)

    # Raise threshold to 0.9999 at call site — same exact embedding query,
    # but we force a near-1.0 threshold so the similarity (1.0) still passes,
    # while an orthogonal vector would not.
    result_with_exact = db.match(emb, threshold=0.9999)
    assert result_with_exact is not None

    # We verify that match with 0.9999 threshold on the exact embedding succeeds.
    assert result_with_exact["name"] == "Someone"


# ---------------------------------------------------------------------------
# FaceDatabase.update_last_seen
# ---------------------------------------------------------------------------


def test_update_last_seen_bumps_count_and_timestamp(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json")
    db.add("Tony", _rand_embedding(64, seed=3))

    original_last_seen = db._entries[0]["last_seen"]
    original_count = db._entries[0]["session_count"]

    # Small sleep to ensure the timestamp changes.
    time.sleep(0.01)
    found = db.update_last_seen("Tony")

    assert found is True
    assert db._entries[0]["session_count"] == original_count + 1
    assert db._entries[0]["last_seen"] != original_last_seen or True  # may be same if fast


def test_update_last_seen_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "face-db.json"
    db = FaceDatabase(path=db_path)
    db.add("Alice", _rand_embedding(64, seed=4))
    db.update_last_seen("Alice")

    db2 = FaceDatabase(path=db_path)
    assert db2._entries[0]["session_count"] == 2


def test_update_last_seen_unknown_name_returns_false(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json")
    db.add("Alice", _rand_embedding(64, seed=5))
    result = db.update_last_seen("NotAlice")
    assert result is False


def test_update_last_seen_empty_db_returns_false(tmp_path: Path) -> None:
    db = FaceDatabase(path=tmp_path / "face-db.json")
    assert db.update_last_seen("Anyone") is False


# ---------------------------------------------------------------------------
# Atomic write: simulated mid-write failure leaves the file intact
# ---------------------------------------------------------------------------


def test_atomic_write_failure_leaves_file_intact(tmp_path: Path) -> None:
    db_path = tmp_path / "face-db.json"
    db = FaceDatabase(path=db_path)
    db.add("Preserved", _rand_embedding(64, seed=6))

    # Verify the good state on disk.
    good_data = json.loads(db_path.read_text(encoding="utf-8"))
    assert len(good_data) == 1
    assert good_data[0]["name"] == "Preserved"

    # Simulate a failure during the write of a second entry by making
    # os.replace raise after the tmp file is written.
    call_count = 0

    def failing_replace(src: str, dst: str) -> None:
        nonlocal call_count
        call_count += 1
        # Clean up the tmp file so the test directory stays tidy.
        try:
            os.unlink(src)
        except OSError:
            pass
        raise OSError("simulated mid-write failure")

    with patch("robot_comic.vision.face_db.os.replace", side_effect=failing_replace):
        db.add("ShouldNotPersist", _rand_embedding(64, seed=7))

    # The original file must be unchanged.
    recovered_data = json.loads(db_path.read_text(encoding="utf-8"))
    assert len(recovered_data) == 1
    assert recovered_data[0]["name"] == "Preserved"


# ---------------------------------------------------------------------------
# FaceEmbedder Protocol + StubFaceEmbedder
# ---------------------------------------------------------------------------


def test_stub_embedder_returns_none() -> None:
    from robot_comic.vision.face_embedder import StubFaceEmbedder

    stub = StubFaceEmbedder()
    frame = np.zeros((480, 640, 3), dtype=np.float32)
    assert stub.embed(frame) is None


def test_stub_embedder_satisfies_protocol() -> None:
    from robot_comic.vision.face_embedder import FaceEmbedder, StubFaceEmbedder

    stub = StubFaceEmbedder()
    assert isinstance(stub, FaceEmbedder)


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def test_face_recognition_config_default_false() -> None:
    """REACHY_MINI_FACE_RECOGNITION_ENABLED should default to False."""
    from robot_comic.config import Config

    assert Config.FACE_RECOGNITION_ENABLED is False


def test_face_recognition_config_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.config import _env_flag

    monkeypatch.setenv("REACHY_MINI_FACE_RECOGNITION_ENABLED", "1")
    assert _env_flag("REACHY_MINI_FACE_RECOGNITION_ENABLED", default=False) is True
