"""Tests for the Roast tool."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

_PROFILE_PATH = Path(__file__).parents[2] / "profiles" / "don_rickles" / "roast.py"


def _load_roast_module():
    spec = importlib.util.spec_from_file_location("don_rickles_roast", _PROFILE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["don_rickles_roast"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def roast_mod():
    return _load_roast_module()


@pytest.fixture
def Roast(roast_mod):
    return roast_mod.Roast


@pytest.fixture
def parse_extraction(roast_mod):
    return roast_mod._parse_extraction


def make_deps(scan_response: str = "PERSON: center", extraction_response: str = "") -> ToolDependencies:
    deps = MagicMock(spec=ToolDependencies)
    deps.motion_duration_s = 0.0
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    deps.camera_worker = MagicMock()
    deps.camera_worker.get_latest_frame.return_value = frame
    deps.vision_processor = MagicMock()
    deps.vision_processor.process_image.side_effect = [scan_response, extraction_response]
    deps.movement_manager = MagicMock()
    deps.reachy_mini = MagicMock()
    deps.reachy_mini.get_current_head_pose.return_value = MagicMock()
    deps.reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    return deps


EXTRACTION_TEXT = (
    "hair: thinning on top, attempting a combover\n"
    "clothing: wrinkled button-down, tucked in badly\n"
    "build: stocky, comfortable with it\n"
    "expression: trying to look relaxed, not pulling it off\n"
    "standout: the mustache. that's the whole story.\n"
    "energy: nervous, fidgety"
)


# --- _parse_extraction ---

def test_parse_extraction_returns_all_fields(parse_extraction):
    result = parse_extraction(EXTRACTION_TEXT)
    assert result["hair"] == "thinning on top, attempting a combover"
    assert result["clothing"] == "wrinkled button-down, tucked in badly"
    assert result["build"] == "stocky, comfortable with it"
    assert result["expression"] == "trying to look relaxed, not pulling it off"
    assert result["standout"] == "the mustache. that's the whole story."
    assert result["energy"] == "nervous, fidgety"


def test_parse_extraction_missing_field_returns_unknown(parse_extraction):
    result = parse_extraction("hair: thinning\nclothing: wrinkled")
    assert result["hair"] == "thinning"
    assert result["clothing"] == "wrinkled"
    assert result["build"] == "unknown"
    assert result["standout"] == "unknown"


# --- Roast tool ---

@pytest.mark.asyncio
async def test_roast_returns_no_subject_when_no_person_detected(Roast):
    deps = make_deps(scan_response="NO PERSON")
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert result == {"no_subject": True}
    assert deps.camera_worker.get_latest_frame.call_count == 1


@pytest.mark.asyncio
async def test_roast_moves_head_left_when_person_on_left(Roast):
    deps = make_deps(scan_response="PERSON: left", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    deps.movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
async def test_roast_moves_head_right_when_person_on_right(Roast):
    deps = make_deps(scan_response="PERSON: right", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    deps.movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
async def test_roast_returns_parsed_fields_on_success(Roast):
    deps = make_deps(scan_response="PERSON: center", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert result["hair"] == "thinning on top, attempting a combover"
    assert result["standout"] == "the mustache. that's the whole story."
    assert result["energy"] == "nervous, fidgety"


@pytest.mark.asyncio
async def test_roast_captures_two_frames(Roast):
    deps = make_deps(scan_response="PERSON: center", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    assert deps.camera_worker.get_latest_frame.call_count == 2


@pytest.mark.asyncio
async def test_roast_returns_error_when_camera_worker_unavailable(Roast):
    deps = make_deps()
    deps.camera_worker = None
    result = await Roast()(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_roast_returns_error_when_no_frame_available(Roast):
    deps = make_deps()
    deps.camera_worker.get_latest_frame.return_value = None
    result = await Roast()(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_roast_falls_back_to_b64_when_no_vision_processor(Roast):
    deps = make_deps()
    deps.vision_processor = None
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert "b64_scene" in result
    assert "extraction_prompt" in result
