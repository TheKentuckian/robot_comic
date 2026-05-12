"""Tests for the Gemini 429 retry/backoff helpers and TTS retry-after handling."""

from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot_comic.gemini_retry import (
    compute_backoff,
    is_rate_limit_error,
    describe_quota_failure,
    extract_retry_after_seconds,
)
from robot_comic.tools.core_tools import ToolDependencies


def _make_429(retry_delay: str | None = "23s", header_retry_after: str | None = None):
    """Build a google-genai ClientError shaped like a real 429 from the Gemini API."""
    from google.genai.errors import ClientError

    details: list[dict] = []
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    details.append(
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": ("generativelanguage.googleapis.com/generate_content_free_tier_requests")}],
        }
    )
    response_json = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded",
            "details": details,
        }
    }
    response = None
    if header_retry_after is not None:
        response = MagicMock()
        response.headers = {"Retry-After": header_retry_after}
    return ClientError(429, response_json, response)


def test_is_rate_limit_error_detects_429() -> None:
    exc = _make_429()
    assert is_rate_limit_error(exc) is True


def test_is_rate_limit_error_ignores_500() -> None:
    assert is_rate_limit_error(RuntimeError("503 UNAVAILABLE")) is False


def test_extract_retry_after_from_retry_info() -> None:
    exc = _make_429(retry_delay="42s")
    assert extract_retry_after_seconds(exc) == pytest.approx(42.0)


def test_extract_retry_after_prefers_http_header() -> None:
    exc = _make_429(retry_delay="42s", header_retry_after="7")
    # Header takes precedence over RetryInfo.
    assert extract_retry_after_seconds(exc) == pytest.approx(7.0)


def test_extract_retry_after_handles_missing_info() -> None:
    exc = _make_429(retry_delay=None)
    assert extract_retry_after_seconds(exc) is None


def test_describe_quota_failure_extracts_metric() -> None:
    exc = _make_429()
    assert "free_tier" in describe_quota_failure(exc)


def test_compute_backoff_honours_retry_after() -> None:
    # Jitter is uniform in [0, jitter_s], so result is in [retry_after, retry_after + jitter_s].
    delay = compute_backoff(attempt=0, base_delay=1.0, retry_after_s=5.0, jitter_s=0.0)
    assert delay == pytest.approx(5.0)


def test_compute_backoff_caps_long_retry_after() -> None:
    delay = compute_backoff(attempt=0, base_delay=1.0, retry_after_s=3600.0, cap_s=30.0, jitter_s=0.0)
    assert delay == pytest.approx(30.0)


def test_compute_backoff_exponential_without_hint() -> None:
    d0 = compute_backoff(attempt=0, base_delay=1.0, retry_after_s=None, jitter_s=0.0)
    d1 = compute_backoff(attempt=1, base_delay=1.0, retry_after_s=None, jitter_s=0.0)
    d2 = compute_backoff(attempt=2, base_delay=1.0, retry_after_s=None, jitter_s=0.0)
    assert d0 == pytest.approx(1.0)
    assert d1 == pytest.approx(2.0)
    assert d2 == pytest.approx(4.0)


def _make_handler():
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = GeminiTTSResponseHandler(deps)
    handler._client = MagicMock()
    return handler


@pytest.mark.asyncio
async def test_tts_429_backs_off_using_retry_after() -> None:
    """A 429 with Retry-After=2 should sleep ~2s before the next attempt."""
    handler = _make_handler()

    err = _make_429(retry_delay="2s")
    call_count = 0

    async def flaky_generate(model, contents, config):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise err
        # 2nd attempt succeeds with valid base64 audio
        import base64

        encoded = base64.b64encode(b"\x00" * 480).decode()
        part = MagicMock()
        part.inline_data.data = encoded
        candidate = MagicMock()
        candidate.content.parts = [part]
        resp = MagicMock()
        resp.candidates = [candidate]
        return resp

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = flaky_generate

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    with patch("robot_comic.gemini_tts.asyncio.sleep", new=fake_sleep):
        result = await handler._call_tts_with_retry("hello")

    assert result is not None
    assert call_count == 2
    assert sleep_calls, "expected the retry-after path to sleep at least once"
    # The first (and only) sleep should be >= the suggested retry-after seconds
    # (some jitter is allowed on top).
    assert sleep_calls[0] >= 2.0
    assert sleep_calls[0] <= 2.0 + 1.0  # 2s + max jitter (0.5s) + headroom


@pytest.mark.asyncio
async def test_tts_429_exhaustion_returns_none_and_logs_loudly(caplog) -> None:
    """All three attempts hit 429 → returns None and logs an ERROR mentioning the quota."""
    import logging

    handler = _make_handler()
    err = _make_429(retry_delay="1s")

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = AsyncMock(side_effect=err)

    async def no_sleep(delay: float) -> None:
        return None

    caplog.set_level(logging.ERROR, logger="robot_comic.gemini_tts")
    with patch("robot_comic.gemini_tts.asyncio.sleep", new=no_sleep):
        result = await handler._call_tts_with_retry("hello")

    assert result is None
    # Loud final log line, not silent
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR and "exhausted" in r.message]
    assert error_records, "expected an ERROR log line on exhaustion"
    # The quota descriptor surfaces in the message.
    assert any("free_tier" in r.message or "quota=" in r.message for r in error_records)


@pytest.mark.asyncio
async def test_llama_gemini_tts_429_backs_off() -> None:
    """LlamaGeminiTTS handler also honours Retry-After on 429."""
    from robot_comic.llama_gemini_tts import LlamaGeminiTTSResponseHandler

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LlamaGeminiTTSResponseHandler(deps)
    handler._client = MagicMock()

    err = _make_429(retry_delay="3s")
    call_count = 0

    async def flaky_generate(model, contents, config):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise err
        import base64

        encoded = base64.b64encode(b"\x00" * 240).decode()
        part = MagicMock()
        part.inline_data.data = encoded
        candidate = MagicMock()
        candidate.content.parts = [part]
        resp = MagicMock()
        resp.candidates = [candidate]
        return resp

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = flaky_generate

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    with patch("robot_comic.llama_gemini_tts.asyncio.sleep", new=fake_sleep):
        result = await handler._call_gemini_tts("hello")

    assert result is not None
    assert sleep_calls and sleep_calls[0] >= 3.0
