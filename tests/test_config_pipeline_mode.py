"""Tests for the PIPELINE_MODE config dial (Phase 0 of the pipeline refactor).

PIPELINE_MODE distinguishes 3-phase composable pipelines (moonshine STT + any
LLM + any TTS) from bundled speech-to-speech backends (OpenAI Realtime, Gemini
Live, HF Realtime) where STT/LLM/TTS are fused into one websocket session.

It's a 4th dial alongside the existing STT (AUDIO_INPUT_BACKEND), LLM
(LLM_BACKEND), and TTS (AUDIO_OUTPUT_BACKEND) knobs. Values:

- ``composable`` (default) — the other three dials decide the pipeline
- ``openai_realtime`` — bundled, ignores the other dials
- ``gemini_live`` — bundled, ignores the other dials
- ``hf_realtime`` — bundled, ignores the other dials

Backwards-compat: when ``REACHY_MINI_PIPELINE_MODE`` is unset, the value is
derived from the resolved ``(AUDIO_INPUT_BACKEND, AUDIO_OUTPUT_BACKEND)``
combo so existing ``.env`` files keep working without operator action.
"""

from __future__ import annotations
import importlib

import pytest


@pytest.fixture(autouse=True)
def _restore_config_object():
    """Save and restore the config singleton (mirrors test_config_new_flags)."""
    cfg_mod = importlib.import_module("robot_comic.config")
    original = cfg_mod.config
    yield
    cfg_mod.config = original


def _reload_config(monkeypatch, env: dict):
    # Strip leftover audio-pipeline env vars from prior tests in the suite —
    # ``_persist_env_values`` in console.py mutates ``os.environ`` directly,
    # which pytest's monkeypatch fixture does NOT auto-rollback.
    for stale in (
        "REACHY_MINI_PIPELINE_MODE",
        "REACHY_MINI_AUDIO_INPUT_BACKEND",
        "REACHY_MINI_AUDIO_OUTPUT_BACKEND",
        "REACHY_MINI_LLM_BACKEND",
    ):
        if stale not in env:
            monkeypatch.delenv(stale, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import robot_comic.config as cfg_mod

    importlib.reload(cfg_mod)
    return cfg_mod


# ---------------------------------------------------------------------------
# derive_pipeline_mode — pure function
# ---------------------------------------------------------------------------


def test_derive_pipeline_mode_bundled_openai_realtime():
    from robot_comic.config import (
        AUDIO_INPUT_OPENAI_REALTIME,
        AUDIO_OUTPUT_OPENAI_REALTIME,
        PIPELINE_MODE_OPENAI_REALTIME,
        derive_pipeline_mode,
    )

    assert (
        derive_pipeline_mode(AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME)
        == PIPELINE_MODE_OPENAI_REALTIME
    )


def test_derive_pipeline_mode_bundled_gemini_live():
    from robot_comic.config import (
        AUDIO_INPUT_GEMINI_LIVE,
        AUDIO_OUTPUT_GEMINI_LIVE,
        PIPELINE_MODE_GEMINI_LIVE,
        derive_pipeline_mode,
    )

    assert derive_pipeline_mode(AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE) == PIPELINE_MODE_GEMINI_LIVE


def test_derive_pipeline_mode_bundled_hf_realtime():
    from robot_comic.config import (
        AUDIO_INPUT_HF,
        AUDIO_OUTPUT_HF,
        PIPELINE_MODE_HF_REALTIME,
        derive_pipeline_mode,
    )

    assert derive_pipeline_mode(AUDIO_INPUT_HF, AUDIO_OUTPUT_HF) == PIPELINE_MODE_HF_REALTIME


def test_derive_pipeline_mode_composable_with_moonshine():
    """Moonshine STT always means composable, regardless of output backend."""
    from robot_comic.config import (
        AUDIO_INPUT_MOONSHINE,
        AUDIO_OUTPUT_CHATTERBOX,
        AUDIO_OUTPUT_ELEVENLABS,
        AUDIO_OUTPUT_GEMINI_TTS,
        PIPELINE_MODE_COMPOSABLE,
        derive_pipeline_mode,
    )

    for output in (AUDIO_OUTPUT_CHATTERBOX, AUDIO_OUTPUT_ELEVENLABS, AUDIO_OUTPUT_GEMINI_TTS):
        assert derive_pipeline_mode(AUDIO_INPUT_MOONSHINE, output) == PIPELINE_MODE_COMPOSABLE, (
            f"moonshine + {output} should be composable"
        )


def test_derive_pipeline_mode_mismatched_input_output_falls_back_to_composable():
    """Inputs that don't match any bundled pair (e.g. moonshine STT routed to an
    openai_realtime_output TTS) are valid composable pipelines."""
    from robot_comic.config import (
        AUDIO_INPUT_MOONSHINE,
        PIPELINE_MODE_COMPOSABLE,
        AUDIO_OUTPUT_OPENAI_REALTIME,
        derive_pipeline_mode,
    )

    assert derive_pipeline_mode(AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_OPENAI_REALTIME) == PIPELINE_MODE_COMPOSABLE


# ---------------------------------------------------------------------------
# Config singleton — explicit env var
# ---------------------------------------------------------------------------


def test_pipeline_mode_env_var_takes_priority(monkeypatch):
    """``REACHY_MINI_PIPELINE_MODE`` overrides the derivation when set."""
    cfg = _reload_config(
        monkeypatch,
        {
            "REACHY_MINI_PIPELINE_MODE": "openai_realtime",
            # Make derivation point at composable so we can prove the override won.
            "REACHY_MINI_AUDIO_INPUT_BACKEND": "moonshine",
            "REACHY_MINI_AUDIO_OUTPUT_BACKEND": "elevenlabs",
        },
    )
    assert cfg.config.PIPELINE_MODE == "openai_realtime"


def test_pipeline_mode_defaults_to_derived_when_env_unset(monkeypatch):
    """Without an explicit env var, PIPELINE_MODE is derived from input/output."""
    monkeypatch.delenv("REACHY_MINI_PIPELINE_MODE", raising=False)
    cfg = _reload_config(
        monkeypatch,
        {
            "REACHY_MINI_AUDIO_INPUT_BACKEND": "moonshine",
            "REACHY_MINI_AUDIO_OUTPUT_BACKEND": "elevenlabs",
        },
    )
    # moonshine + elevenlabs is a composable pipeline.
    assert cfg.config.PIPELINE_MODE == "composable"


def test_pipeline_mode_derived_openai_realtime_when_inputs_match(monkeypatch):
    """The (openai_realtime_input, openai_realtime_output) pair derives to the bundled mode."""
    monkeypatch.delenv("REACHY_MINI_PIPELINE_MODE", raising=False)
    cfg = _reload_config(
        monkeypatch,
        {
            "REACHY_MINI_AUDIO_INPUT_BACKEND": "openai_realtime_input",
            "REACHY_MINI_AUDIO_OUTPUT_BACKEND": "openai_realtime_output",
        },
    )
    assert cfg.config.PIPELINE_MODE == "openai_realtime"


def test_pipeline_mode_derived_gemini_live_when_inputs_match(monkeypatch):
    monkeypatch.delenv("REACHY_MINI_PIPELINE_MODE", raising=False)
    cfg = _reload_config(
        monkeypatch,
        {
            "REACHY_MINI_AUDIO_INPUT_BACKEND": "gemini_live_input",
            "REACHY_MINI_AUDIO_OUTPUT_BACKEND": "gemini_live_output",
        },
    )
    assert cfg.config.PIPELINE_MODE == "gemini_live"


def test_pipeline_mode_derived_hf_realtime_when_inputs_match(monkeypatch):
    monkeypatch.delenv("REACHY_MINI_PIPELINE_MODE", raising=False)
    cfg = _reload_config(
        monkeypatch,
        {
            "REACHY_MINI_AUDIO_INPUT_BACKEND": "hf_input",
            "REACHY_MINI_AUDIO_OUTPUT_BACKEND": "hf_output",
        },
    )
    assert cfg.config.PIPELINE_MODE == "hf_realtime"


def test_pipeline_mode_invalid_env_value_falls_back_to_derived(monkeypatch, caplog):
    """An unknown PIPELINE_MODE value logs a warning and falls back to derivation."""
    monkeypatch.setenv("REACHY_MINI_PIPELINE_MODE", "not-a-real-mode")
    cfg = _reload_config(
        monkeypatch,
        {
            "REACHY_MINI_AUDIO_INPUT_BACKEND": "moonshine",
            "REACHY_MINI_AUDIO_OUTPUT_BACKEND": "chatterbox",
        },
    )
    # Falls back to derivation (moonshine + chatterbox → composable).
    assert cfg.config.PIPELINE_MODE == "composable"


def test_pipeline_mode_choices_constant_exposed():
    """``PIPELINE_MODE_CHOICES`` lists the canonical values."""
    from robot_comic.config import (
        PIPELINE_MODE_CHOICES,
        PIPELINE_MODE_COMPOSABLE,
        PIPELINE_MODE_GEMINI_LIVE,
        PIPELINE_MODE_HF_REALTIME,
        PIPELINE_MODE_OPENAI_REALTIME,
    )

    assert set(PIPELINE_MODE_CHOICES) == {
        PIPELINE_MODE_COMPOSABLE,
        PIPELINE_MODE_OPENAI_REALTIME,
        PIPELINE_MODE_GEMINI_LIVE,
        PIPELINE_MODE_HF_REALTIME,
    }
