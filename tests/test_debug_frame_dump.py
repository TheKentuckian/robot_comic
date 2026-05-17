"""Tests for the POST /api/debug/dump-camera-frame admin route (#269).

Covers:
- Success path: camera returns a frame → PNG written, response ok=True + path.
- No-frame path: camera returns None → 503 ok=False.
- No-handler path: LocalStream built without a handler (sim mode) → 503.
- model_selection=0 audit: _detect_face passes model_selection=0 to MediaPipe.
- FACE_DETECTION_CONFIDENCE config attribute is present and defaults to 0.3.
"""

from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import robot_comic.tools.greet as greet_mod
from robot_comic.console import LocalStream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame() -> np.ndarray:  # type: ignore[type-arg]
    """Return a tiny valid BGR frame."""
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _make_client_with_camera(frame: "np.ndarray | None") -> TestClient:  # type: ignore[type-arg]
    """Build a TestClient whose handler.deps.camera_worker returns *frame*."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = frame

    deps = SimpleNamespace(camera_worker=camera_worker)
    handler = MagicMock()
    handler.deps = deps

    app = FastAPI()
    stream = LocalStream(handler, robot=None, settings_app=app)
    stream.init_admin_ui()
    return TestClient(app)


def _make_client_no_handler() -> TestClient:
    """Build a TestClient with handler=None (sim/settings-only mode)."""
    app = FastAPI()
    stream = LocalStream(handler=None, robot=None, settings_app=app)
    stream.init_admin_ui()
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/debug/dump-camera-frame — success path
# ---------------------------------------------------------------------------


class TestDumpCameraFrameSuccess:
    def test_returns_ok_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful dump must return {ok: true} with HTTP 200."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_with_camera(_blank_frame())

        with patch("robot_comic.camera_frame_encoding.encode_bgr_frame_as_png", return_value=b"PNG"):
            resp = client.post("/api/debug/dump-camera-frame")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_response_contains_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The response path must point to a .png file."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_with_camera(_blank_frame())

        with patch("robot_comic.camera_frame_encoding.encode_bgr_frame_as_png", return_value=b"PNG"):
            resp = client.post("/api/debug/dump-camera-frame")

        data = resp.json()
        assert "path" in data
        assert data["path"].endswith(".png")

    def test_png_file_written_to_disk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A PNG file must actually be created at the returned path."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_with_camera(_blank_frame())

        fake_png = b"\x89PNG\r\n\x1a\n"  # minimal PNG header bytes
        with patch("robot_comic.camera_frame_encoding.encode_bgr_frame_as_png", return_value=fake_png):
            resp = client.post("/api/debug/dump-camera-frame")

        data = resp.json()
        written = Path(data["path"])
        assert written.exists(), f"Expected PNG at {written}"
        assert written.read_bytes() == fake_png

    def test_path_contains_logs_debug_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dump path must be under logs/debug/ (project convention)."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_with_camera(_blank_frame())

        with patch("robot_comic.camera_frame_encoding.encode_bgr_frame_as_png", return_value=b"PNG"):
            resp = client.post("/api/debug/dump-camera-frame")

        data = resp.json()
        # Path may be absolute or relative; normalise for comparison.
        norm = Path(data["path"]).as_posix()
        assert "logs/debug" in norm, f"Expected 'logs/debug' in {norm!r}"


# ---------------------------------------------------------------------------
# /api/debug/dump-camera-frame — no-frame path
# ---------------------------------------------------------------------------


class TestDumpCameraFrameNoFrame:
    def test_returns_503_when_frame_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """camera_worker.get_latest_frame() returning None → 503."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_with_camera(None)

        resp = client.post("/api/debug/dump-camera-frame")

        assert resp.status_code == 503
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data

    def test_returns_503_when_no_handler(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When handler is None (sim mode), camera is unavailable → 503."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_no_handler()

        resp = client.post("/api/debug/dump-camera-frame")

        assert resp.status_code == 503
        data = resp.json()
        assert data["ok"] is False

    def test_no_file_written_on_no_frame(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No PNG should be written when the camera returns None."""
        monkeypatch.chdir(tmp_path)
        client = _make_client_with_camera(None)

        client.post("/api/debug/dump-camera-frame")

        debug_dir = tmp_path / "logs" / "debug"
        files = list(debug_dir.glob("*.png")) if debug_dir.exists() else []
        assert files == [], f"Unexpected files: {files}"


# ---------------------------------------------------------------------------
# model_selection audit (#269)
# ---------------------------------------------------------------------------


class TestModelSelectionAudit:
    """Verify _detect_face passes model_selection=0 to MediaPipe FaceDetection."""

    def test_detect_face_uses_short_range_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_detect_face must construct FaceDetection with model_selection=0."""
        monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)

        fake_detector = MagicMock()
        fake_detector.__enter__ = MagicMock(return_value=fake_detector)
        fake_detector.__exit__ = MagicMock(return_value=False)
        fake_detector.process.return_value = SimpleNamespace(detections=[])

        fake_face_detection = MagicMock(return_value=fake_detector)

        fake_mp = SimpleNamespace(FaceDetection=fake_face_detection)
        monkeypatch.setattr(greet_mod, "_mp_face_detection", fake_mp)

        frame = _blank_frame()
        greet_mod._detect_face(frame)

        fake_face_detection.assert_called_once()
        _, kwargs = fake_face_detection.call_args
        assert kwargs.get("model_selection") == 0, (
            f"Expected model_selection=0 but got {kwargs.get('model_selection')!r}. "
            "Short-range model is correct for Reachy Mini ~50 cm use case."
        )

    def test_detect_face_with_scores_uses_short_range_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_detect_face_with_scores must also use model_selection=0."""
        monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)

        fake_detector = MagicMock()
        fake_detector.__enter__ = MagicMock(return_value=fake_detector)
        fake_detector.__exit__ = MagicMock(return_value=False)
        fake_detector.process.return_value = SimpleNamespace(detections=[])

        fake_face_detection = MagicMock(return_value=fake_detector)
        fake_mp = SimpleNamespace(FaceDetection=fake_face_detection)
        monkeypatch.setattr(greet_mod, "_mp_face_detection", fake_mp)

        frame = _blank_frame()
        greet_mod._detect_face_with_scores(frame)

        fake_face_detection.assert_called_once()
        _, kwargs = fake_face_detection.call_args
        assert kwargs.get("model_selection") == 0


# ---------------------------------------------------------------------------
# FACE_DETECTION_CONFIDENCE in config (#269)
# ---------------------------------------------------------------------------


class TestFaceDetectionConfidenceConfig:
    def test_config_has_face_detection_confidence_attribute(self) -> None:
        """config.FACE_DETECTION_CONFIDENCE must exist and be a float."""
        from robot_comic.config import config

        assert hasattr(config, "FACE_DETECTION_CONFIDENCE"), (
            "config.FACE_DETECTION_CONFIDENCE is missing; add it to _RuntimeConfig and refresh."
        )
        assert isinstance(config.FACE_DETECTION_CONFIDENCE, float)

    def test_config_default_is_point_three(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default value must be 0.3 (per #269 spec)."""
        monkeypatch.delenv("REACHY_MINI_FACE_DETECTION_CONFIDENCE", raising=False)
        from robot_comic import config as config_module

        config_module.refresh_runtime_config_from_env()
        assert config_module.config.FACE_DETECTION_CONFIDENCE == pytest.approx(0.3)

    def test_config_reads_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh_runtime_config_from_env re-reads the env var."""
        monkeypatch.setenv("REACHY_MINI_FACE_DETECTION_CONFIDENCE", "0.6")
        from robot_comic import config as config_module

        config_module.refresh_runtime_config_from_env()
        assert config_module.config.FACE_DETECTION_CONFIDENCE == pytest.approx(0.6)

    def test_config_clamps_above_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values above 1.0 must be clamped to 1.0."""
        monkeypatch.setenv("REACHY_MINI_FACE_DETECTION_CONFIDENCE", "9.9")
        from robot_comic import config as config_module

        config_module.refresh_runtime_config_from_env()
        assert config_module.config.FACE_DETECTION_CONFIDENCE == pytest.approx(1.0)

    def test_config_clamps_below_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values below 0.0 must be clamped to 0.0."""
        monkeypatch.setenv("REACHY_MINI_FACE_DETECTION_CONFIDENCE", "-0.5")
        from robot_comic import config as config_module

        config_module.refresh_runtime_config_from_env()
        assert config_module.config.FACE_DETECTION_CONFIDENCE == pytest.approx(0.0)
