"""Tests for the 3-column STT/LLM/TTS pipeline picker route changes (#245).

Server-side only (no headless browser). Tests cover:
- POST /backend_config with llm_backend persists both output and LLM env vars.
- Unsupported combination (Gemini LLM + OpenAI Realtime output) → 400.
- Back-compat: POST without llm_backend defaults to "llama".
"""

from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot_comic.config import (
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
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
    backend_provider: str = "local_stt",
    openai_api_key: str = "sk-test",
    gemini_api_key: str = "AIza-test",
    elevenlabs_api_key: str = "el-test",
    llm_backend: str = LLM_BACKEND_LLAMA,
) -> TestClient:
    monkeypatch.setattr(config, "BACKEND_PROVIDER", backend_provider)
    monkeypatch.setattr(config, "OPENAI_API_KEY", openai_api_key)
    monkeypatch.setattr(config, "GEMINI_API_KEY", gemini_api_key)
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", elevenlabs_api_key)
    monkeypatch.setattr(config, "LLM_BACKEND", llm_backend)
    monkeypatch.setattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")
    monkeypatch.setattr(config, "LOCAL_STT_LANGUAGE", "en")
    monkeypatch.setattr(config, "LOCAL_STT_MODEL", "tiny_streaming")
    monkeypatch.setattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35)
    monkeypatch.setenv("BACKEND_PROVIDER", backend_provider)
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
            "backend": "local_stt",
            "local_stt_response_backend": "chatterbox",
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
    assert "LOCAL_STT_RESPONSE_BACKEND=chatterbox" in env_text
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
            "backend": "local_stt",
            "local_stt_response_backend": "elevenlabs",
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
    assert "LOCAL_STT_RESPONSE_BACKEND=elevenlabs" in env_text
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
            "backend": "local_stt",
            "api_key": "sk-test",
            "local_stt_response_backend": "openai",
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
            "backend": "local_stt",
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 8765,
            "local_stt_response_backend": "huggingface",
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
            "backend": "local_stt",
            "api_key": "sk-test",
            "local_stt_response_backend": "openai",
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
