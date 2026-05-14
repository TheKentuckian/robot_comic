"""Tests for FaceRecognitionEmbedder — concrete face embedding backend."""

from __future__ import annotations
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame(h: int = 64, w: int = 64) -> "np.ndarray[np.uint8, np.dtype[np.uint8]]":
    """Return a small blank RGB uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fake_face_recognition_module() -> ModuleType:
    """Build a minimal fake ``face_recognition`` module."""
    mod = ModuleType("face_recognition")
    mod.face_locations = MagicMock(return_value=[])  # type: ignore[attr-defined]
    mod.face_encodings = MagicMock(return_value=[])  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# ImportError when library is missing
# ---------------------------------------------------------------------------


def test_embedder_raises_import_error_when_library_missing() -> None:
    """Constructing the embedder without face_recognition installed must raise ImportError."""
    # Remove face_recognition from sys.modules and block the import.
    sys.modules.pop("face_recognition", None)

    with patch.dict("sys.modules", {"face_recognition": None}):  # type: ignore[dict-item]
        from robot_comic.vision.face_recognition_embedder import FaceRecognitionEmbedder

        with pytest.raises(ImportError, match="face_recognition"):
            FaceRecognitionEmbedder()


# ---------------------------------------------------------------------------
# Returns None when no face is detected
# ---------------------------------------------------------------------------


def test_embedder_returns_none_when_no_face_detected() -> None:
    """embed() returns None when face_locations returns an empty list."""
    fake_fr = _fake_face_recognition_module()
    fake_fr.face_locations = MagicMock(return_value=[])  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"face_recognition": fake_fr}):
        # Reload to pick up the patched module.
        import importlib

        import robot_comic.vision.face_recognition_embedder as _mod

        importlib.reload(_mod)
        embedder = _mod.FaceRecognitionEmbedder()
        result = embedder.embed(_blank_frame())

    assert result is None


# ---------------------------------------------------------------------------
# Returns 128-element array when a face is found
# ---------------------------------------------------------------------------


def test_embedder_returns_128d_array_when_face_found() -> None:
    """embed() returns a (128,) float array when face_locations returns a location."""
    fake_embedding = np.random.default_rng(42).standard_normal(128)
    fake_fr = _fake_face_recognition_module()
    # One face location: (top, right, bottom, left)
    fake_fr.face_locations = MagicMock(return_value=[(10, 50, 50, 10)])  # type: ignore[attr-defined]
    fake_fr.face_encodings = MagicMock(return_value=[fake_embedding])  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"face_recognition": fake_fr}):
        import importlib

        import robot_comic.vision.face_recognition_embedder as _mod

        importlib.reload(_mod)
        embedder = _mod.FaceRecognitionEmbedder()
        result = embedder.embed(_blank_frame())

    assert result is not None
    assert result.shape == (128,)
    assert np.allclose(result, fake_embedding)


# ---------------------------------------------------------------------------
# Picks the largest face when multiple are detected
# ---------------------------------------------------------------------------


def test_embedder_picks_largest_face() -> None:
    """When multiple faces are present, embed() selects the largest bounding box."""
    # small_face: 10×10 = 100 px²; large_face: 40×40 = 1600 px²
    small_face = (10, 20, 20, 10)  # top, right, bottom, left → 10×10
    large_face = (5, 45, 45, 5)  # top, right, bottom, left → 40×40

    emb_small = np.zeros(128)
    emb_large = np.ones(128)

    fake_fr = _fake_face_recognition_module()
    fake_fr.face_locations = MagicMock(return_value=[small_face, large_face])  # type: ignore[attr-defined]

    def _encodings(frame: object, locations: list) -> list:  # type: ignore[type-arg]
        # Return a different embedding depending on which location was passed.
        if locations == [large_face]:
            return [emb_large]
        return [emb_small]

    fake_fr.face_encodings = MagicMock(side_effect=_encodings)  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"face_recognition": fake_fr}):
        import importlib

        import robot_comic.vision.face_recognition_embedder as _mod

        importlib.reload(_mod)
        embedder = _mod.FaceRecognitionEmbedder()
        result = embedder.embed(_blank_frame(100, 100))

    assert result is not None
    assert np.allclose(result, emb_large), "Should have used the largest face's embedding"


# ---------------------------------------------------------------------------
# Returns None when face_encodings returns empty (edge case)
# ---------------------------------------------------------------------------


def test_embedder_returns_none_when_encodings_empty() -> None:
    """embed() returns None when face_encodings returns an empty list despite a location."""
    fake_fr = _fake_face_recognition_module()
    fake_fr.face_locations = MagicMock(return_value=[(10, 50, 50, 10)])  # type: ignore[attr-defined]
    fake_fr.face_encodings = MagicMock(return_value=[])  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"face_recognition": fake_fr}):
        import importlib

        import robot_comic.vision.face_recognition_embedder as _mod

        importlib.reload(_mod)
        embedder = _mod.FaceRecognitionEmbedder()
        result = embedder.embed(_blank_frame())

    assert result is None
