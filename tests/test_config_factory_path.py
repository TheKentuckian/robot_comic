"""Phase 4b: REACHY_MINI_FACTORY_PATH dial wiring (#337).

Covers the normaliser, the Config field, and the env-var refresh hook for the
new ``FACTORY_PATH`` dial that gates the composable factory path.
"""

from __future__ import annotations
import logging

import pytest

from robot_comic import config as cfg


def test_constants_defined() -> None:
    assert cfg.FACTORY_PATH_ENV == "REACHY_MINI_FACTORY_PATH"
    assert cfg.FACTORY_PATH_LEGACY == "legacy"
    assert cfg.FACTORY_PATH_COMPOSABLE == "composable"
    assert cfg.FACTORY_PATH_CHOICES == ("legacy", "composable")
    assert cfg.DEFAULT_FACTORY_PATH == "legacy"


def test_normalize_default_when_unset() -> None:
    assert cfg._normalize_factory_path(None) == "legacy"
    assert cfg._normalize_factory_path("") == "legacy"
    assert cfg._normalize_factory_path("   ") == "legacy"


def test_normalize_known_values() -> None:
    assert cfg._normalize_factory_path("composable") == "composable"
    assert cfg._normalize_factory_path("COMPOSABLE") == "composable"
    assert cfg._normalize_factory_path("  legacy  ") == "legacy"


def test_normalize_invalid_falls_back_to_legacy_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
        result = cfg._normalize_factory_path("hybrid")
    assert result == "legacy"
    assert any("hybrid" in record.message for record in caplog.records)
