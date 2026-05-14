"""Tests for the face-recognition path in greet(action='identify')."""

from __future__ import annotations
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


_GREET_PATH = Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "greet.py"


def _load_greet_module():
    """Load greet.py from its package path using importlib."""
    spec = importlib.util.spec_from_file_location("greet_face_test", _GREET_PATH)
    assert spec and spec.loader, f"Cannot load module from {_GREET_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["greet_face_test"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def greet_mod():
    """Module-scoped fixture: loaded greet module."""
    return _load_greet_module()


@pytest.fixture
def Greet(greet_mod):
    """The Greet class."""
    return greet_mod.Greet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_embedding(seed: int = 0) -> "np.ndarray[np.float64, np.dtype[np.float64]]":
    """Return a random unit-length 128-D float64 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float64)
    return v / np.linalg.norm(v)


def _blank_frame() -> "np.ndarray[np.uint8, np.dtype[np.uint8]]":
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _make_deps(
    *,
    face_recognition_enabled: bool = True,
    face_embedder: object = None,
    face_db: object = None,
    camera_frame: "np.ndarray | None" = None,
) -> MagicMock:
    """Build a minimal mock ToolDependencies."""
    deps = MagicMock()
    deps.motion_duration_s = 0.0
    deps.face_embedder = face_embedder
    deps.face_db = face_db

    camera = MagicMock()
    camera.get_latest_frame.return_value = camera_frame if camera_frame is not None else _blank_frame()
    deps.camera_worker = camera

    # Patch config via the module attribute so the greet module sees it.
    return deps


# ---------------------------------------------------------------------------
# Patch helper — set FACE_RECOGNITION_ENABLED on the config singleton used by
# the greet module, without touching the real config.
# ---------------------------------------------------------------------------


def _patch_config(greet_mod, enabled: bool):
    """Context-manager-like: returns a patch object for the config used in greet_mod."""
    import unittest.mock as mock

    fake_config = MagicMock()
    fake_config.config.FACE_RECOGNITION_ENABLED = enabled
    return mock.patch.object(greet_mod, "_config", fake_config)


# ---------------------------------------------------------------------------
# Face recognition ENABLED — face matches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_face_match_returns_recalled_name(Greet, greet_mod) -> None:
    """Face-db match returns returning=True with the recalled name."""
    embedding = _rand_embedding(seed=1)

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = embedding

    fake_db = MagicMock()
    fake_db.match.return_value = {
        "name": "Tony",
        "last_seen": "2026-05-14T12:00:00+00:00",
        "session_count": 3,
    }

    deps = _make_deps(face_recognition_enabled=True, face_embedder=fake_embedder, face_db=fake_db)

    with _patch_config(greet_mod, enabled=True):
        result = await Greet()(deps, action="identify")

    assert result["returning"] is True
    assert result["name"] == "Tony"
    assert result.get("face_match") is True
    fake_embedder.embed.assert_called_once()
    fake_db.match.assert_called_once()


# ---------------------------------------------------------------------------
# Face recognition ENABLED — no face match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_no_face_match_returns_name_none(Greet, greet_mod) -> None:
    """When face is detected but DB has no match, returning=False and pending_embedding present."""
    embedding = _rand_embedding(seed=2)

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = embedding

    fake_db = MagicMock()
    fake_db.match.return_value = None  # no match

    deps = _make_deps(face_recognition_enabled=True, face_embedder=fake_embedder, face_db=fake_db)

    with _patch_config(greet_mod, enabled=True):
        result = await Greet()(deps, action="identify")

    assert result["returning"] is False
    assert result["name"] is None
    # pending_embedding should be a list (serialisable for LLM)
    assert isinstance(result.get("pending_embedding"), list)
    assert len(result["pending_embedding"]) == 128


# ---------------------------------------------------------------------------
# Face recognition ENABLED — no face detected in frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_no_face_in_frame_returns_pending_true(Greet, greet_mod) -> None:
    """When embedder returns None (no face), pending_embedding=True is returned."""
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = None  # no face detected

    fake_db = MagicMock()

    deps = _make_deps(face_recognition_enabled=True, face_embedder=fake_embedder, face_db=fake_db)

    with _patch_config(greet_mod, enabled=True):
        result = await Greet()(deps, action="identify")

    assert result["returning"] is False
    assert result["name"] is None
    assert result.get("pending_embedding") is True


# ---------------------------------------------------------------------------
# Face recognition DISABLED — falls back to name error when no name given
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_disabled_falls_back_to_name_error(Greet, greet_mod, tmp_path) -> None:
    """When face recognition is disabled, identify without a name returns an error."""
    deps = _make_deps(face_recognition_enabled=False, face_embedder=None, face_db=None)

    with _patch_config(greet_mod, enabled=False):
        result = await Greet(session_dir=tmp_path)(deps, action="identify")

    assert "error" in result


# ---------------------------------------------------------------------------
# Face recognition DISABLED — name-based path works normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_disabled_name_path_returns_not_found(Greet, greet_mod, tmp_path) -> None:
    """When face recognition is disabled and no session file exists, returns returning=False."""
    session_dir = tmp_path / ".comedy_sessions"
    session_dir.mkdir()

    deps = _make_deps(face_recognition_enabled=False, face_embedder=None, face_db=None)

    with _patch_config(greet_mod, enabled=False):
        result = await Greet(session_dir=session_dir)(deps, action="identify", name="Tony")

    assert result == {"returning": False, "name_received": "Tony"}


# ---------------------------------------------------------------------------
# Face recognition ENABLED but embedder/db are None (misconfigured)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_enabled_but_deps_none_falls_back_to_name(Greet, greet_mod, tmp_path) -> None:
    """When enabled but face_embedder/face_db are None, falls back to name-based path."""
    session_dir = tmp_path / ".comedy_sessions"
    session_dir.mkdir()

    deps = _make_deps(face_recognition_enabled=True, face_embedder=None, face_db=None)

    with _patch_config(greet_mod, enabled=True):
        # face_embedder is None → face path returns None → name-based fallback
        result = await Greet(session_dir=session_dir)(deps, action="identify", name="Bob")

    # No sessions → returning=False with name_received
    assert result == {"returning": False, "name_received": "Bob"}
