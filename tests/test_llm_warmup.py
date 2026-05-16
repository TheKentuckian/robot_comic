"""Unit tests for the LLM KV-cache warmup fired at startup.

The warmup sends a single /v1/chat/completions request with the active system
prompt and max_tokens=1 so llama-server primes its KV cache before the first
real user turn.  All tests mock httpx — no real server call is made.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler():
    """Return a ChatterboxTTSResponseHandler with a pre-attached mock HTTP client."""
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    from robot_comic.tools.core_tools import ToolDependencies

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = ChatterboxTTSResponseHandler(deps)
    handler._http = AsyncMock()
    return handler


def _make_streaming_mock(chunks: list[bytes] | None = None) -> AsyncMock:
    """Build an async context-manager mock that iterates *chunks* via aiter_bytes."""
    chunks = chunks or [b"data: [DONE]\n\n"]

    async def _aiter_bytes():
        for chunk in chunks:
            yield chunk

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_bytes = _aiter_bytes

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# 1. Warmup enabled — POST is made and INFO logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_enabled_posts_and_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    """With feature enabled and a successful mock POST, warmup completes and logs INFO."""
    handler = _make_handler()
    stream_cm = _make_streaming_mock()
    handler._http.stream = MagicMock(return_value=stream_cm)

    with (
        patch("robot_comic.chatterbox_tts.config") as mock_config,
        patch("robot_comic.chatterbox_tts.get_session_instructions", return_value="You are a comedian."),
        patch("robot_comic.chatterbox_tts.get_active_tool_specs", return_value=[]),
    ):
        mock_config.LLM_WARMUP_ENABLED = True
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"

        with caplog.at_level(logging.DEBUG, logger="robot_comic.chatterbox_tts"):
            await handler._warmup_llm_kv_cache()

    # POST must have been issued
    handler._http.stream.assert_called_once()
    call_kwargs = handler._http.stream.call_args
    assert call_kwargs.args[0] == "POST"
    assert "/v1/chat/completions" in call_kwargs.args[1]

    # INFO lines for begin and complete
    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("llm warmup" in m and "begin" in m for m in info_messages)
    assert any("llm warmup" in m and "complete" in m for m in info_messages)

    # No warnings
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. Warmup disabled via env — no POST made
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_disabled_skips_post(caplog: pytest.LogCaptureFixture) -> None:
    """With REACHY_MINI_LLM_WARMUP_ENABLED=0, no POST is made."""
    handler = _make_handler()
    handler._http.stream = MagicMock()

    with patch("robot_comic.chatterbox_tts.config") as mock_config:
        mock_config.LLM_WARMUP_ENABLED = False
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True

        with caplog.at_level(logging.DEBUG, logger="robot_comic.chatterbox_tts"):
            await handler._warmup_llm_kv_cache()

    handler._http.stream.assert_not_called()
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. Health check disabled — warmup skipped automatically
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_skipped_when_health_check_disabled(caplog: pytest.LogCaptureFixture) -> None:
    """When LLAMA_HEALTH_CHECK_ENABLED=False, warmup is skipped regardless of LLM_WARMUP_ENABLED."""
    handler = _make_handler()
    handler._http.stream = MagicMock()

    with patch("robot_comic.chatterbox_tts.config") as mock_config:
        mock_config.LLM_WARMUP_ENABLED = True
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = False

        with caplog.at_level(logging.DEBUG, logger="robot_comic.chatterbox_tts"):
            await handler._warmup_llm_kv_cache()

    handler._http.stream.assert_not_called()
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. POST fails — logs WARNING and does NOT raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_failure_logs_warning_and_continues(caplog: pytest.LogCaptureFixture) -> None:
    """When the POST raises, warmup logs WARNING and continues without raising."""
    handler = _make_handler()

    import httpx

    failing_cm = AsyncMock()
    failing_cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
    failing_cm.__aexit__ = AsyncMock(return_value=False)
    handler._http.stream = MagicMock(return_value=failing_cm)

    with (
        patch("robot_comic.chatterbox_tts.config") as mock_config,
        patch("robot_comic.chatterbox_tts.get_session_instructions", return_value="system"),
        patch("robot_comic.chatterbox_tts.get_active_tool_specs", return_value=[]),
    ):
        mock_config.LLM_WARMUP_ENABLED = True
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"

        with caplog.at_level(logging.WARNING, logger="robot_comic.chatterbox_tts"):
            # Must not raise
            await handler._warmup_llm_kv_cache()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "warmup" in warnings[0].message.lower()


# ---------------------------------------------------------------------------
# 5. System prompt is included in the warmup payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_payload_contains_system_prompt(caplog: pytest.LogCaptureFixture) -> None:
    """The warmup request body must include a system-role message with the active prompt."""
    handler = _make_handler()
    captured_payload: dict = {}  # type: ignore[type-arg]

    def _capture_stream(method: str, url: str, **kwargs: object) -> AsyncMock:
        captured_payload.update(kwargs.get("json", {}))  # type: ignore[arg-type]
        return _make_streaming_mock()

    handler._http.stream = MagicMock(side_effect=_capture_stream)

    expected_prompt = "You are Don Rickles the comedian robot."

    with (
        patch("robot_comic.chatterbox_tts.config") as mock_config,
        patch("robot_comic.chatterbox_tts.get_session_instructions", return_value=expected_prompt),
        patch("robot_comic.chatterbox_tts.get_active_tool_specs", return_value=[]),
    ):
        mock_config.LLM_WARMUP_ENABLED = True
        mock_config.LLAMA_HEALTH_CHECK_ENABLED = True
        mock_config.LLAMA_CPP_URL = "http://localhost:8080"

        await handler._warmup_llm_kv_cache()

    messages = captured_payload.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert system_messages, "Expected at least one system-role message in warmup payload"
    assert system_messages[0]["content"] == expected_prompt
