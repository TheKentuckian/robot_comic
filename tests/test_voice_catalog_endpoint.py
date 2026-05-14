"""Tests for the GET /api/voices/catalog admin endpoint (#304)."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import robot_comic.elevenlabs_voices as voices_mod
from robot_comic.console import LocalStream


def _make_client() -> TestClient:
    """Build a TestClient backed by a minimal LocalStream with admin UI."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), None, settings_app=app)
    stream.init_admin_ui()
    return TestClient(app)


def _reset_voice_cache() -> None:
    """Drop the module-level caches so each test starts from a clean slate."""
    voices_mod._voice_cache = None
    voices_mod._voice_records_cache = None


class TestVoiceCatalogEndpoint:
    """Schema, error, and contract checks for /api/voices/catalog."""

    def setup_method(self) -> None:
        _reset_voice_cache()

    def teardown_method(self) -> None:
        _reset_voice_cache()

    def test_returns_503_when_elevenlabs_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ELEVENLABS_API_KEY -> 503 with documented error body, never 500."""
        from robot_comic.config import config

        monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "", raising=False)

        client = _make_client()
        resp = client.get("/api/voices/catalog")

        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body
        assert "elevenlabs" in body["error"].lower()

    def test_returns_200_with_expected_schema_when_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cached catalog is returned as a list of {voice_id, name, category}."""
        from robot_comic.config import config

        monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "fake-key", raising=False)

        # Pre-populate the cache so fetch_elevenlabs_voices() is a no-op and we
        # never touch the live ElevenLabs API in unit tests.
        voices_mod._voice_records_cache = [
            {"name": "Don Rickles (IVC)", "voice_id": "r3TOducz", "category": "cloned"},
            {"name": "Bill - Deep, Resonant", "voice_id": "pqHfBill", "category": "premade"},
        ]
        voices_mod._voice_cache = {
            "Don Rickles (IVC)": "r3TOducz",
            "Bill - Deep, Resonant": "pqHfBill",
        }

        client = _make_client()
        resp = client.get("/api/voices/catalog")

        assert resp.status_code == 200
        body = resp.json()
        assert "voices" in body
        assert isinstance(body["voices"], list)
        assert len(body["voices"]) == 2

        # Every entry must include the three documented fields.
        for entry in body["voices"]:
            assert set(entry.keys()) >= {"voice_id", "name", "category"}

        # Spot-check the actual values mapped through.
        by_id = {v["voice_id"]: v for v in body["voices"]}
        assert by_id["r3TOducz"]["name"] == "Don Rickles (IVC)"
        assert by_id["r3TOducz"]["category"] == "cloned"
        assert by_id["pqHfBill"]["category"] == "premade"

    def test_catalog_entries_have_required_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every voice in the response includes voice_id, name, and category."""
        from robot_comic.config import config

        monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "fake-key", raising=False)

        voices_mod._voice_records_cache = [
            {"name": "Alpha", "voice_id": "id-alpha", "category": "premade"},
            {"name": "Beta", "voice_id": "id-beta", "category": "cloned"},
            {"name": "Gamma", "voice_id": "id-gamma", "category": "generated"},
        ]
        voices_mod._voice_cache = {
            "Alpha": "id-alpha",
            "Beta": "id-beta",
            "Gamma": "id-gamma",
        }

        client = _make_client()
        resp = client.get("/api/voices/catalog")

        assert resp.status_code == 200
        voices = resp.json()["voices"]
        assert len(voices) == 3
        for v in voices:
            assert v["voice_id"], "voice_id must be non-empty"
            assert v["name"], "name must be non-empty"
            assert "category" in v, "category field must be present"

    def test_falls_back_to_hardcoded_when_api_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the API key is set but the upstream fetch yields fallback records,
        the endpoint still returns 200 with the fallback list (not 503)."""
        from robot_comic.config import config

        monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "fake-key", raising=False)

        # Monkeypatch httpx.AsyncClient so the fetch hits the except branch
        # and writes _FALLBACK_VOICES into the cache.
        import httpx

        class _FailingClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> "_FailingClient":
                return self

            async def __aexit__(self, *exc: Any) -> bool:
                return False

            async def get(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("network down")

        monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)

        client = _make_client()
        resp = client.get("/api/voices/catalog")

        assert resp.status_code == 200
        body = resp.json()
        assert "voices" in body
        # Fallback list is non-empty and shaped correctly.
        assert len(body["voices"]) > 0
        for v in body["voices"]:
            assert "voice_id" in v
            assert "name" in v
            assert "category" in v
