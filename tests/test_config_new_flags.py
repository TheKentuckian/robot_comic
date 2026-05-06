import importlib
import os
import pytest


def _reload_config(monkeypatch, env: dict):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import robot_comic.config as cfg_mod
    importlib.reload(cfg_mod)
    return cfg_mod.config


def test_gemini_live_video_streaming_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.GEMINI_LIVE_VIDEO_STREAMING is False


def test_gemini_live_video_streaming_env_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"GEMINI_LIVE_VIDEO_STREAMING": "true"})
    assert cfg.GEMINI_LIVE_VIDEO_STREAMING is True


def test_movement_speed_factor_defaults_0_6(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(0.6)


def test_movement_speed_factor_clamped_high(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOVEMENT_SPEED_FACTOR": "5.0"})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(2.0)


def test_movement_speed_factor_clamped_low(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOVEMENT_SPEED_FACTOR": "0.0"})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(0.1)


def test_moonshine_heartbeat_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.MOONSHINE_HEARTBEAT is False


def test_moonshine_heartbeat_env_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOONSHINE_HEARTBEAT": "true"})
    assert cfg.MOONSHINE_HEARTBEAT is True
