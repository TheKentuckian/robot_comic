"""Tests for the 3-column STT/LLM/TTS pipeline picker route changes (#245).

Server-side only (no headless browser). Tests cover:
- POST /backend_config with llm_backend persists both audio output and LLM env vars.
- Unsupported combination (Gemini LLM + OpenAI Realtime output) → 400.
- Back-compat: POST without llm_backend defaults to "llama".

Phase 4f reshapes the POST payload to send ``pipeline_mode`` +
``audio_input_backend`` + ``audio_output_backend`` instead of the legacy
``backend`` / ``local_stt_response_backend`` fields.  The assertions
follow accordingly.
"""

from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot_comic.config import (
    AUDIO_OUTPUT_HF,
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    PIPELINE_MODE_COMPOSABLE,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    config,
)
from robot_comic.console import LocalStream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pipeline_mode: str = PIPELINE_MODE_COMPOSABLE,
    audio_input_backend: str = AUDIO_INPUT_MOONSHINE,
    audio_output_backend: str = AUDIO_OUTPUT_OPENAI_REALTIME,
    openai_api_key: str = "sk-test",
    gemini_api_key: str = "AIza-test",
    elevenlabs_api_key: str = "el-test",
    llm_backend: str = LLM_BACKEND_LLAMA,
) -> TestClient:
    monkeypatch.setattr(config, "PIPELINE_MODE", pipeline_mode)
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", audio_input_backend)
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", audio_output_backend)
    monkeypatch.setattr(config, "OPENAI_API_KEY", openai_api_key)
    monkeypatch.setattr(config, "GEMINI_API_KEY", gemini_api_key)
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", elevenlabs_api_key)
    monkeypatch.setattr(config, "LLM_BACKEND", llm_backend)
    monkeypatch.setattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")
    monkeypatch.setattr(config, "LOCAL_STT_LANGUAGE", "en")
    monkeypatch.setattr(config, "LOCAL_STT_MODEL", "tiny_streaming")
    monkeypatch.setattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35)
    monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_chatterbox_with_gemini_llm_persists_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST with output=chatterbox + llm_backend=gemini should persist both env vars."""
    client = _make_client(tmp_path, monkeypatch, gemini_api_key="AIza-test")

    resp = client.post(
        "/backend_config",
        json={
            "pipeline_mode": PIPELINE_MODE_COMPOSABLE,
            "audio_input_backend": AUDIO_INPUT_MOONSHINE,
            "audio_output_backend": AUDIO_OUTPUT_CHATTERBOX,
            "llm_backend": LLM_BACKEND_GEMINI,
            "local_stt_cache_dir": "./cache/moonshine_voice",
            "local_stt_language": "en",
            "local_stt_model": "tiny_streaming",
            "local_stt_update_interval": 0.35,
        },
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["ok"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert f"REACHY_MINI_AUDIO_OUTPUT_BACKEND={AUDIO_OUTPUT_CHATTERBOX}" in env_text
    assert f"REACHY_MINI_LLM_BACKEND={LLM_BACKEND_GEMINI}" in env_text


def test_post_elevenlabs_with_gemini_llm_persists_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST with output=elevenlabs + llm_backend=gemini should persist both env vars."""
    client = _make_client(tmp_path, monkeypatch, elevenlabs_api_key="el-test")

    resp = client.post(
        "/backend_config",
        json={
            "pipeline_mode": PIPELINE_MODE_COMPOSABLE,
            "audio_input_backend": AUDIO_INPUT_MOONSHINE,
            "audio_output_backend": AUDIO_OUTPUT_ELEVENLABS,
            "llm_backend": LLM_BACKEND_GEMINI,
            "local_stt_cache_dir": "./cache/moonshine_voice",
            "local_stt_language": "en",
            "local_stt_model": "tiny_streaming",
            "local_stt_update_interval": 0.35,
        },
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["ok"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert f"REACHY_MINI_AUDIO_OUTPUT_BACKEND={AUDIO_OUTPUT_ELEVENLABS}" in env_text
    assert f"REACHY_MINI_LLM_BACKEND={LLM_BACKEND_GEMINI}" in env_text


def test_post_unsupported_gemini_llm_plus_openai_realtime_returns_400(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini text LLM cannot be paired with OpenAI Realtime output — expect 400."""
    client = _make_client(tmp_path, monkeypatch)

    resp = client.post(
        "/backend_config",
        json={
            "pipeline_mode": PIPELINE_MODE_COMPOSABLE,
            "audio_input_backend": AUDIO_INPUT_MOONSHINE,
            "audio_output_backend": AUDIO_OUTPUT_OPENAI_REALTIME,
            "api_key": "sk-test",
            "llm_backend": LLM_BACKEND_GEMINI,
            "local_stt_cache_dir": "./cache/moonshine_voice",
            "local_stt_language": "en",
            "local_stt_model": "tiny_streaming",
            "local_stt_update_interval": 0.35,
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data.get("error") == "unsupported_pipeline_combination"


def test_post_unsupported_gemini_llm_plus_hf_realtime_returns_400(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini text LLM cannot be paired with HuggingFace Realtime output — expect 400."""
    client = _make_client(tmp_path, monkeypatch)
    # Provide HF WS URL so "hf" key check passes
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")

    resp = client.post(
        "/backend_config",
        json={
            "pipeline_mode": PIPELINE_MODE_COMPOSABLE,
            "audio_input_backend": AUDIO_INPUT_MOONSHINE,
            "audio_output_backend": AUDIO_OUTPUT_HF,
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 8765,
            "llm_backend": LLM_BACKEND_GEMINI,
            "local_stt_cache_dir": "./cache/moonshine_voice",
            "local_stt_language": "en",
            "local_stt_model": "tiny_streaming",
            "local_stt_update_interval": 0.35,
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data.get("error") == "unsupported_pipeline_combination"


def test_post_without_llm_backend_defaults_to_llama(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: omitting llm_backend should default to 'llama' (no 400 error)."""
    client = _make_client(tmp_path, monkeypatch)

    resp = client.post(
        "/backend_config",
        json={
            "pipeline_mode": PIPELINE_MODE_COMPOSABLE,
            "audio_input_backend": AUDIO_INPUT_MOONSHINE,
            "audio_output_backend": AUDIO_OUTPUT_OPENAI_REALTIME,
            "api_key": "sk-test",
            # llm_backend intentionally omitted
            "local_stt_cache_dir": "./cache/moonshine_voice",
            "local_stt_language": "en",
            "local_stt_model": "tiny_streaming",
            "local_stt_update_interval": 0.35,
        },
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["ok"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    # Should have written the default LLM_BACKEND
    assert f"REACHY_MINI_LLM_BACKEND={LLM_BACKEND_LLAMA}" in env_text


def test_status_payload_includes_llm_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /status should include llm_backend so the UI can restore the LLM column."""
    client = _make_client(tmp_path, monkeypatch, llm_backend=LLM_BACKEND_GEMINI)

    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm_backend" in data
    assert data["llm_backend"] == LLM_BACKEND_GEMINI


def test_status_payload_drops_legacy_backend_provider_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 4f: status no longer emits the retired dial fields."""
    client = _make_client(tmp_path, monkeypatch)

    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "backend_provider" not in data
    assert "local_stt_response_backend" not in data
    assert "local_stt_response_backend_choices" not in data
    assert data["pipeline_mode"] == PIPELINE_MODE_COMPOSABLE
    assert data["audio_input_backend"] == AUDIO_INPUT_MOONSHINE
    assert data["audio_output_backend"] == AUDIO_OUTPUT_OPENAI_REALTIME


def test_post_rejects_missing_pipeline_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty pipeline_mode field must be rejected — no implicit fallback."""
    client = _make_client(tmp_path, monkeypatch)

    resp = client.post(
        "/backend_config",
        json={
            "pipeline_mode": "",
            "audio_input_backend": AUDIO_INPUT_MOONSHINE,
            "audio_output_backend": AUDIO_OUTPUT_OPENAI_REALTIME,
            "api_key": "sk-test",
        },
    )
    assert resp.status_code == 400
    assert resp.json().get("error") == "invalid_pipeline_mode"
