"""Tests for the LLM-extraction path in joke_history.extract_punchline_via_llm."""

from __future__ import annotations
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from robot_comic.joke_history import last_sentence_of, extract_punchline_via_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_http_client(response_body: dict | str | None = None, *, raise_exc: Exception | None = None) -> MagicMock:
    """Build a mock httpx.AsyncClient whose post() returns the given body or raises."""
    client = MagicMock(spec=httpx.AsyncClient)
    if raise_exc is not None:
        client.post = AsyncMock(side_effect=raise_exc)
        return client

    if isinstance(response_body, dict):
        raw = json.dumps(response_body)
    else:
        raw = response_body or ""

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"choices": [{"message": {"content": raw}}]})
    client.post = AsyncMock(return_value=mock_resp)
    return client


# ---------------------------------------------------------------------------
# Feature-flag gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_returns_heuristic_when_feature_disabled() -> None:
    """When JOKE_HISTORY_LLM_EXTRACT_ENABLED=False the heuristic is used directly."""
    text = "You're a real gem. Cubic zirconia, but still!"
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=AssertionError("should not be called"))

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = False
        result = await extract_punchline_via_llm(text, client)

    assert result is not None
    assert result["punchline"] == last_sentence_of(text)
    assert result["topic"] == ""


# ---------------------------------------------------------------------------
# Happy-path: valid JSON response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_parses_valid_json_response() -> None:
    """Returns the LLM-supplied punchline and topic when response is valid JSON."""
    response = {"punchline": "Cubic zirconia, but still!", "topic": "appearance"}
    client = _make_http_client(response)

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm("Some assistant text.", client)

    assert result is not None
    assert result["punchline"] == "Cubic zirconia, but still!"
    assert result["topic"] == "appearance"


@pytest.mark.asyncio
async def test_extract_returns_none_punchline_for_setup_or_banter() -> None:
    """When LLM returns {\"punchline\": null, ...} the result dict has punchline=None."""
    response = {"punchline": None, "topic": "banter"}
    client = _make_http_client(response)

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm("Just some setup text.", client)

    assert result is not None
    assert result["punchline"] is None


@pytest.mark.asyncio
async def test_extract_strips_markdown_fences() -> None:
    """JSON wrapped in markdown code fences is still parsed correctly."""
    inner = {"punchline": "Nice one!", "topic": "compliment"}
    raw = "```json\n" + json.dumps(inner) + "\n```"
    client = _make_http_client(raw)

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm("assistant text", client)

    assert result is not None
    assert result["punchline"] == "Nice one!"


# ---------------------------------------------------------------------------
# Fallback: network errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_falls_back_on_timeout() -> None:
    """Falls back to last_sentence_of on httpx.TimeoutException."""
    text = "Your suit looks great. Did you buy it at Sears?"
    client = _make_http_client(raise_exc=httpx.TimeoutException("timed out"))

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm(text, client)

    assert result is not None
    assert result["punchline"] == last_sentence_of(text)
    assert result["topic"] == ""


@pytest.mark.asyncio
async def test_extract_falls_back_on_network_error() -> None:
    """Falls back to last_sentence_of on httpx.NetworkError."""
    text = "This is a network error test."
    client = _make_http_client(raise_exc=httpx.NetworkError("conn refused"))

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm(text, client)

    assert result is not None
    assert result["punchline"] == last_sentence_of(text)


@pytest.mark.asyncio
async def test_extract_falls_back_on_connect_error() -> None:
    """Falls back to last_sentence_of on httpx.ConnectError."""
    text = "Testing connect error fallback."
    client = _make_http_client(raise_exc=httpx.ConnectError("no route"))

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm(text, client)

    assert result is not None
    assert result["punchline"] == last_sentence_of(text)


# ---------------------------------------------------------------------------
# Fallback: non-JSON / malformed responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_falls_back_on_non_json_response() -> None:
    """Falls back to last_sentence_of when LLM returns prose instead of JSON."""
    text = "A classic punchline right here."
    client = _make_http_client("Sorry, I can't help with that.")

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm(text, client)

    assert result is not None
    assert result["punchline"] == last_sentence_of(text)
    assert result["topic"] == ""


@pytest.mark.asyncio
async def test_extract_falls_back_on_json_array_instead_of_dict() -> None:
    """Falls back when LLM returns a JSON array rather than a dict."""
    text = "Array fallback test."
    client = _make_http_client("[1, 2, 3]")

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        result = await extract_punchline_via_llm(text, client)

    assert result is not None
    assert result["punchline"] == last_sentence_of(text)


# ---------------------------------------------------------------------------
# Timeout is honoured (the passed timeout is forwarded to the HTTP call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_passes_timeout_to_post() -> None:
    """The 500 ms timeout is forwarded as a per-request timeout to client.post()."""
    from robot_comic.joke_history import _EXTRACT_TIMEOUT_S

    response = {"punchline": "Quick!", "topic": "speed"}
    client = _make_http_client(response)

    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.JOKE_HISTORY_LLM_EXTRACT_ENABLED = True
        await extract_punchline_via_llm("assistant text", client)

    # Verify that client.post was called with a timeout argument.
    call_kwargs = client.post.call_args
    assert call_kwargs is not None
    timeout_arg = call_kwargs.kwargs.get("timeout")
    assert timeout_arg is not None
    # The timeout should reflect our _EXTRACT_TIMEOUT_S constant.
    if hasattr(timeout_arg, "read"):
        # httpx.Timeout object
        assert timeout_arg.read == pytest.approx(_EXTRACT_TIMEOUT_S)
    else:
        assert float(timeout_arg) == pytest.approx(_EXTRACT_TIMEOUT_S)
