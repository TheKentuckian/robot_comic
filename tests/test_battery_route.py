"""Tests for the /api/battery route and _read_battery helper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import robot_comic.console as console_mod
from robot_comic.console import LocalStream, _read_battery


def _make_client(robot=None):
    """Build a TestClient backed by a minimal LocalStream with admin UI."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream.init_admin_ui()
    return TestClient(app)


def _clear_cache():
    """Reset the module-level battery cache and warning flag between tests."""
    console_mod._battery_cache.clear()
    console_mod._battery_warned_once = False


# ---------------------------------------------------------------------------
# /api/battery route
# ---------------------------------------------------------------------------


class TestBatteryRoute:
    def setup_method(self):
        _clear_cache()

    def test_sim_mode_returns_sim_source(self):
        """When robot is None (sim/headless-without-robot), source must be 'sim'."""
        client = _make_client(robot=None)
        resp = client.get("/api/battery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "sim"
        assert data["percent"] is None

    def test_robot_without_battery_attr_returns_unknown(self):
        """Robot object with no .battery attribute -> source 'unknown'."""
        robot = SimpleNamespace()  # no battery attribute at all
        client = _make_client(robot=robot)
        resp = client.get("/api/battery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "unknown"
        assert data["percent"] is None

    def test_robot_with_battery_attr_returns_robot_source(self):
        """Robot with a .battery object -> source 'robot' with correct fields."""
        battery = SimpleNamespace(percent=78, voltage=7.4, charging=False)
        robot = SimpleNamespace(battery=battery)
        client = _make_client(robot=robot)
        resp = client.get("/api/battery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "robot"
        assert data["percent"] == 78
        assert data["voltage"] == pytest.approx(7.4)
        assert data["charging"] is False

    def test_robot_with_charging_battery(self):
        """charging=True propagates correctly."""
        battery = SimpleNamespace(percent=55, voltage=7.6, charging=True)
        robot = SimpleNamespace(battery=battery)
        client = _make_client(robot=robot)
        resp = client.get("/api/battery")
        data = resp.json()
        assert data["source"] == "robot"
        assert data["charging"] is True

    def test_response_shape_always_has_source_key(self):
        """Every response must include a 'source' key regardless of mode."""
        for robot in (
            None,
            SimpleNamespace(),
            SimpleNamespace(battery=SimpleNamespace(percent=50, voltage=7.5, charging=False)),
        ):
            _clear_cache()
            client = _make_client(robot=robot)
            resp = client.get("/api/battery")
            assert "source" in resp.json(), f"Missing 'source' for robot={robot!r}"


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestBatteryCache:
    def setup_method(self):
        _clear_cache()

    def test_cache_hit_returns_identical_result_within_ttl(self):
        """Two reads within the TTL window must return the same dict without re-reading."""
        call_count = 0

        class CountingBattery:
            @property
            def percent(self):
                nonlocal call_count
                call_count += 1
                return 42

            voltage = 7.3
            charging = False

        robot = SimpleNamespace(battery=CountingBattery())

        first = _read_battery(robot)
        second = _read_battery(robot)

        assert first == second
        # The property should only have been accessed once (cached on second call).
        assert call_count == 1, f"Expected 1 SDK read, got {call_count}"

    def test_cache_expires_after_ttl(self):
        """After the TTL elapses the cache entry is refreshed."""
        battery = SimpleNamespace(percent=30, voltage=7.2, charging=False)
        robot = SimpleNamespace(battery=battery)

        first = _read_battery(robot)

        # Wind the cache timestamp back past the TTL
        entry = console_mod._battery_cache["entry"]
        entry["ts"] = entry["ts"] - (console_mod._BATTERY_CACHE_TTL_S + 1.0)

        battery.percent = 50  # mutate the underlying value
        second = _read_battery(robot)

        assert first["percent"] == 30
        assert second["percent"] == 50

    def test_sim_mode_cached(self):
        """Sim-mode (robot=None) result is also cached."""
        first = _read_battery(None)
        second = _read_battery(None)
        assert first == second
        assert first["source"] == "sim"

    def test_two_route_calls_within_ttl_hit_cache(self):
        """Two HTTP calls within 5 s return identical JSON bytes (cache hit)."""
        battery = SimpleNamespace(percent=66, voltage=7.5, charging=False)
        robot = SimpleNamespace(battery=battery)
        _clear_cache()

        client = _make_client(robot=robot)
        resp1 = client.get("/api/battery")
        resp2 = client.get("/api/battery")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.content == resp2.content
