"""Shared test helpers for robot_comic test suite."""

from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock


def make_stream_response(lines: list[str]) -> MagicMock:
    """Return an async-context-manager mock suitable for ``_http.stream(...)`` tests.

    The mock's ``aiter_lines()`` yields each line in *lines* in order.

    Usage::

        http_mock.stream = MagicMock(return_value=make_stream_response([
            'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            "data: [DONE]",
        ]))
    """
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    response.raise_for_status = MagicMock()

    async def aiter_lines():
        for line in lines:
            yield line

    response.aiter_lines = aiter_lines
    return response
