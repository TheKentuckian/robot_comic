"""Tests for greet.py's lazy MediaPipe loader (#323 lever 2).

``import mediapipe`` is ~1.2s on the Pi. Greet defers it to first use via
``_check_mp_available()``. These tests assert:

- The lazy checker short-circuits when ``MP_AVAILABLE`` was already set
  (preserves existing test ergonomics — many existing greet tests
  ``monkeypatch.setattr(greet, "MP_AVAILABLE", True)``).
- On first call with ``MP_AVAILABLE=None``, the checker attempts the
  import and caches the result.
- ImportError sets ``MP_AVAILABLE=False`` and clears the detector module.
"""

from __future__ import annotations
import sys
from typing import Any

import pytest

import robot_comic.tools.greet as greet_mod


@pytest.fixture(autouse=True)
def _reset_mp_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with ``MP_AVAILABLE = None`` (un-checked)."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", None)
    monkeypatch.setattr(greet_mod, "_mp_face_detection", None)


def test_check_mp_available_short_circuits_when_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    assert greet_mod._check_mp_available() is True


def test_check_mp_available_short_circuits_when_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", False)
    assert greet_mod._check_mp_available() is False
    # Detector module stays None when MP_AVAILABLE was explicitly False
    assert greet_mod._mp_face_detection is None


def test_check_mp_available_imports_on_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module-level ``MP_AVAILABLE=None`` triggers an import attempt."""
    # Install a fake mediapipe in sys.modules so the import resolves
    fake_mp = type(sys)("mediapipe")
    fake_solutions = type(sys)("mediapipe.solutions")
    setattr(fake_solutions, "face_detection", "<fake face_detection>")
    setattr(fake_mp, "solutions", fake_solutions)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    assert greet_mod._check_mp_available() is True
    assert greet_mod.MP_AVAILABLE is True
    assert greet_mod._mp_face_detection == "<fake face_detection>"


def test_check_mp_available_falls_back_on_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When mediapipe is unavailable, the checker caches MP_AVAILABLE=False."""

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _block_mediapipe(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mediapipe":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_mediapipe)
    # Ensure no stale mediapipe entry survives across tests
    sys.modules.pop("mediapipe", None)

    assert greet_mod._check_mp_available() is False
    assert greet_mod.MP_AVAILABLE is False
    assert greet_mod._mp_face_detection is None


def test_check_mp_available_caches_after_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call returns the cached result without re-importing."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setattr(greet_mod, "_mp_face_detection", "<cached>")
    # No mediapipe in sys.modules — if the checker tried to import, it would
    # raise. The assertion below verifies it short-circuits on the cached
    # boolean instead.
    sys.modules.pop("mediapipe", None)
    assert greet_mod._check_mp_available() is True


def test_detect_face_returns_false_when_mp_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", False)
    # Frame shape doesn't matter — early-return before any frame access
    assert greet_mod._detect_face(object()) is False


def test_detect_face_with_scores_returns_empty_when_mp_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", False)
    found, scores = greet_mod._detect_face_with_scores(object())
    assert found is False
    assert scores == []
