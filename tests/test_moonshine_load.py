"""Tests for Moonshine .ort model preference, fallback, and page-cache prewarm."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import patch

from robot_comic.local_stt_realtime import prewarm_model_file, resolve_ort_model_path


# ---------------------------------------------------------------------------
# resolve_ort_model_path — file-based model paths
# ---------------------------------------------------------------------------


def test_prefers_ort_over_onnx_when_both_present(tmp_path: Path) -> None:
    """When an .ort sibling exists next to the .onnx file, it should be chosen."""
    onnx_file = tmp_path / "model.onnx"
    ort_file = tmp_path / "model.ort"
    onnx_file.write_bytes(b"fake-onnx")
    ort_file.write_bytes(b"fake-ort")

    resolved, fmt = resolve_ort_model_path(onnx_file)

    assert resolved == ort_file
    assert fmt == "ort"


def test_falls_back_to_onnx_when_ort_absent(tmp_path: Path) -> None:
    """When no .ort file exists, the original .onnx path should be returned."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"fake-onnx")

    resolved, fmt = resolve_ort_model_path(onnx_file)

    assert resolved == onnx_file
    assert fmt == "onnx"


# ---------------------------------------------------------------------------
# resolve_ort_model_path — directory-based model paths
# ---------------------------------------------------------------------------


def test_prefers_ort_in_directory_when_present(tmp_path: Path) -> None:
    """For a directory path, .ort files inside should be preferred."""
    (tmp_path / "encoder.onnx").write_bytes(b"fake-onnx")
    ort_file = tmp_path / "encoder.ort"
    ort_file.write_bytes(b"fake-ort")

    resolved, fmt = resolve_ort_model_path(tmp_path)

    assert resolved == ort_file
    assert fmt == "ort"


def test_falls_back_to_directory_when_only_onnx_in_dir(tmp_path: Path) -> None:
    """For a directory path with only .onnx files, directory itself is returned."""
    (tmp_path / "encoder.onnx").write_bytes(b"fake-onnx")

    resolved, fmt = resolve_ort_model_path(tmp_path)

    assert resolved == tmp_path
    assert fmt == "onnx"


def test_returns_directory_unchanged_when_no_model_files(tmp_path: Path) -> None:
    """An empty directory should be returned as-is with format 'onnx'."""
    resolved, fmt = resolve_ort_model_path(tmp_path)

    assert resolved == tmp_path
    assert fmt == "onnx"


# ---------------------------------------------------------------------------
# prewarm_model_file — page-cache prewarm
# ---------------------------------------------------------------------------


def test_prewarm_reads_file_on_non_windows(tmp_path: Path) -> None:
    """On non-Windows platforms, prewarm should open and read the model file."""
    model_file = tmp_path / "model.ort"
    model_file.write_bytes(b"A" * 4096)

    open_calls: list[tuple] = []
    read_calls: list[tuple] = []
    close_calls: list[int] = []

    _FAKE_FD = 42

    def fake_open(path: str, flags: int) -> int:
        open_calls.append((path, flags))
        return _FAKE_FD

    read_sequence = [b"A" * 4096, b""]  # one chunk then EOF

    def fake_read(fd: int, size: int) -> bytes:
        read_calls.append((fd, size))
        return read_sequence.pop(0)

    def fake_close(fd: int) -> None:
        close_calls.append(fd)

    with (
        patch("robot_comic.local_stt_realtime.sys") as mock_sys,
        patch("robot_comic.local_stt_realtime.os.open", side_effect=fake_open),
        patch("robot_comic.local_stt_realtime.os.read", side_effect=fake_read),
        patch("robot_comic.local_stt_realtime.os.close", side_effect=fake_close),
    ):
        mock_sys.platform = "linux"
        prewarm_model_file(model_file)

    assert len(open_calls) == 1
    assert open_calls[0][0] == str(model_file)
    assert len(read_calls) == 2  # one real read, then EOF sentinel
    assert close_calls == [_FAKE_FD]


def test_prewarm_is_noop_on_windows(tmp_path: Path) -> None:
    """On Windows, prewarm should do nothing (not open or read any file)."""
    model_file = tmp_path / "model.ort"
    model_file.write_bytes(b"A" * 4096)

    with (
        patch("robot_comic.local_stt_realtime.sys") as mock_sys,
        patch("robot_comic.local_stt_realtime.os.open") as mock_open,
        patch("robot_comic.local_stt_realtime.os.read") as mock_read,
    ):
        mock_sys.platform = "win32"
        prewarm_model_file(model_file)

    mock_open.assert_not_called()
    mock_read.assert_not_called()


def test_prewarm_reads_ort_file_in_directory(tmp_path: Path) -> None:
    """For a directory, prewarm should read the first .ort file it finds."""
    ort_file = tmp_path / "encoder.ort"
    ort_file.write_bytes(b"B" * 2048)

    open_calls: list[str] = []
    read_sequence = [b"B" * 2048, b""]

    def fake_open(path: str, flags: int) -> int:
        open_calls.append(path)
        return 99

    def fake_read(fd: int, size: int) -> bytes:
        return read_sequence.pop(0)

    def fake_close(fd: int) -> None:
        pass

    with (
        patch("robot_comic.local_stt_realtime.sys") as mock_sys,
        patch("robot_comic.local_stt_realtime.os.open", side_effect=fake_open),
        patch("robot_comic.local_stt_realtime.os.read", side_effect=fake_read),
        patch("robot_comic.local_stt_realtime.os.close", side_effect=fake_close),
    ):
        mock_sys.platform = "linux"
        prewarm_model_file(tmp_path)

    assert len(open_calls) == 1
    assert open_calls[0] == str(ort_file)
