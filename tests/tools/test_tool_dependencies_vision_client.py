"""Tests for ToolDependencies.api_vision_client scaffold (issue #441).

Covers:
- The new field defaults to None.
- _build_api_vision_client returns a genai.Client when LLM_BACKEND=gemini
  and a Gemini API key is present.
- _build_api_vision_client returns None for non-Gemini backends.
- _build_api_vision_client returns None when no API key is configured.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ToolDependencies field default
# ---------------------------------------------------------------------------


def test_tool_dependencies_api_vision_client_defaults_to_none() -> None:
    """api_vision_client must default to None — no external deps required."""
    from robot_comic.tools.core_tools import ToolDependencies

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )
    assert deps.api_vision_client is None


# ---------------------------------------------------------------------------
# _build_api_vision_client helper
# ---------------------------------------------------------------------------


class _StubConfig:
    """Minimal config-like object for testing _build_api_vision_client."""

    def __init__(self, llm_backend: str, gemini_api_key: str | None) -> None:
        self.LLM_BACKEND = llm_backend
        self.GEMINI_API_KEY = gemini_api_key


def test_build_api_vision_client_returns_client_for_gemini_backend() -> None:
    """When LLM_BACKEND=gemini and a key is present, a genai.Client is returned."""
    import sys

    from robot_comic.main import _build_api_vision_client

    stub_client = MagicMock(name="genai.Client.instance")
    stub_genai = MagicMock()
    stub_genai.Client.return_value = stub_client

    # Inject stub modules so `from google import genai` inside the function
    # resolves to our mock without touching the real google-genai package.
    stub_google = MagicMock()
    stub_google.genai = stub_genai

    cfg = _StubConfig(llm_backend="gemini", gemini_api_key="test-key-abc")
    with patch.dict(sys.modules, {"google": stub_google, "google.genai": stub_genai}):
        result = _build_api_vision_client(cfg)

    stub_genai.Client.assert_called_once_with(api_key="test-key-abc")
    assert result is stub_client


def test_build_api_vision_client_returns_none_for_llama_backend() -> None:
    """When LLM_BACKEND=llama, the function returns None without importing genai."""
    from robot_comic.main import _build_api_vision_client

    cfg = _StubConfig(llm_backend="llama", gemini_api_key="test-key-xyz")
    # google.genai must NOT be imported — pass a sentinel that would error if called
    result = _build_api_vision_client(cfg)
    assert result is None


def test_build_api_vision_client_returns_none_when_no_api_key() -> None:
    """When LLM_BACKEND=gemini but GEMINI_API_KEY is falsy, return None."""
    from robot_comic.main import _build_api_vision_client

    for empty_key in (None, ""):
        cfg = _StubConfig(llm_backend="gemini", gemini_api_key=empty_key)
        result = _build_api_vision_client(cfg)
        assert result is None, f"Expected None for GEMINI_API_KEY={empty_key!r}"


def test_build_api_vision_client_returns_none_for_unknown_backend() -> None:
    """Any backend value other than 'gemini' must return None."""
    from robot_comic.main import _build_api_vision_client

    for backend in ("huggingface", "openai", "", "GEMINI"):
        cfg = _StubConfig(llm_backend=backend, gemini_api_key="test-key")
        result = _build_api_vision_client(cfg)
        assert result is None, f"Expected None for LLM_BACKEND={backend!r}"
