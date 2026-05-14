"""Regression test for issue #266 — GET /openapi.json returned HTTP 500.

`ApplyPayload` in :mod:`robot_comic.headless_personality_ui` was defined inside
``mount_personality_routes`` while the module used ``from __future__ import
annotations``. FastAPI evaluated the route annotation ``payload: ApplyPayload |
None = None`` as a forward-ref string, which pydantic could not resolve because
the class was in function-local scope, raising ``PydanticUserError`` during
OpenAPI schema generation.

This test asserts that the admin app can produce its OpenAPI document so the
exact failure mode does not reappear.
"""

from __future__ import annotations
import json
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot_comic.headless_personality_ui import mount_personality_routes


def _build_app() -> FastAPI:
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None)
    return app


def test_openapi_schema_generates_without_pydantic_user_error() -> None:
    """Calling ``app.openapi()`` must not raise once routes are mounted."""
    app = _build_app()
    schema = app.openapi()
    assert isinstance(schema, dict)
    # The /personalities/apply route is the one that referenced ApplyPayload.
    assert "/personalities/apply" in schema.get("paths", {})


def test_openapi_json_endpoint_returns_200() -> None:
    """Hitting /openapi.json through the ASGI app should return JSON, not 500."""
    app = _build_app()
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200, response.text
    payload = json.loads(response.text)
    assert payload.get("openapi"), "expected an OpenAPI version in the document"
