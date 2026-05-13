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


def make_stream_response_with_error(lines: list[str], raise_at_index: int, exc: Exception) -> MagicMock:
    """Return a stream mock that raises *exc* after yielding lines[:raise_at_index].

    Lines at indices 0 .. raise_at_index-1 are yielded normally; on the
    iteration that would yield ``lines[raise_at_index]`` the exception is
    raised instead.  Lines beyond that index are never reached.

    Usage::

        import httpx
        http_mock.stream = MagicMock(
            return_value=make_stream_response_with_error(
                [
                    'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
                    "data: [DONE]",
                ],
                raise_at_index=1,
                exc=httpx.ReadError("connection dropped"),
            )
        )
    """
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    response.raise_for_status = MagicMock()

    async def aiter_lines():
        for i, line in enumerate(lines):
            if i == raise_at_index:
                raise exc
            yield line

    response.aiter_lines = aiter_lines
    return response
