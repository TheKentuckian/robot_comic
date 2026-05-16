"""Unit tests for the llama-server startup health probe."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from robot_comic.tools.core_tools import ToolDependencies


def _make_handler():
    """Create a ChatterboxTTSResponseHandler with a mocked HTTP client."""
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = ChatterboxTTSResponseHandler(deps)
    handler._http = AsyncMock()
    return handler


# ---------------------------------------------------------------------------
# Probe succeeds (HTTP 200) — no warning should be logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_probe_success_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    """A 200 response logs INFO and does NOT emit a WARNING."""
    handler = _make_handler()
    mock_response = MagicMock()
    mock_response.status_code = 200
    handler._http.get = AsyncMock(return_value=mock_response)

    with patch("robot_comic.chatterbox_tts.config") as mock_config:
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True

        with caplog.at_level(logging.DEBUG, logger="robot_comic.chatterbox_tts"):
            await handler._probe_llama_health()

    handler._http.get.assert_awaited_once_with("http://localhost:8080/health", timeout=3.0)
    assert any("health OK" in r.message for r in caplog.records if r.levelno == logging.INFO)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Probe fails with non-200 — WARNING logged with URL and status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_probe_non_200_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A non-200 response logs a WARNING that includes the URL and status code."""
    handler = _make_handler()
    mock_response = MagicMock()
    mock_response.status_code = 503
    handler._http.get = AsyncMock(return_value=mock_response)

    with patch("robot_comic.chatterbox_tts.config") as mock_config:
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True
        mock_config.WOL_MAC = None  # Disable WoL so only one warning fires.

        with caplog.at_level(logging.WARNING, logger="robot_comic.chatterbox_tts"):
            await handler._probe_llama_health()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "localhost:8080" in warnings[0].message
    assert "503" in warnings[0].message


# ---------------------------------------------------------------------------
# Probe times out — WARNING logged with URL and timeout reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_probe_timeout_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A timeout exception logs a WARNING that includes the URL and the exception."""
    handler = _make_handler()
    handler._http.get = AsyncMock(side_effect=httpx.TimeoutException("timed out", request=MagicMock()))

    with patch("robot_comic.chatterbox_tts.config") as mock_config:
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True
        mock_config.WOL_MAC = None  # Disable WoL so only one warning fires.

        with caplog.at_level(logging.WARNING, logger="robot_comic.chatterbox_tts"):
            await handler._probe_llama_health()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "localhost:8080" in warnings[0].message


# ---------------------------------------------------------------------------
# Probe disabled via REACHY_MINI_LLAMA_HEALTH_CHECK=0 — skipped entirely
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_probe_disabled_skips_entirely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When REACHY_MINI_LLAMA_HEALTH_CHECK=0 the probe is never called."""
    handler = _make_handler()
    handler._http.get = AsyncMock()

    with patch("robot_comic.chatterbox_tts.config") as mock_config:
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = False

        with caplog.at_level(logging.DEBUG, logger="robot_comic.chatterbox_tts"):
            await handler._probe_llama_health()

    handler._http.get.assert_not_awaited()
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
