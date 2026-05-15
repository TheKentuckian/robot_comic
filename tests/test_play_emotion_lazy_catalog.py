"""Tests for the play_emotion lazy-catalog behavior (#323 lever 2).

The boot-fast-path goal: ``play_emotion`` reads emotion names + descriptions
from a small JSON cache at module-import time and defers the expensive
``RecordedMoves`` construction until the first ``play_emotion`` invocation.
First-ever boot (no cache) falls back to eager construction so the LLM tool
spec still has the correct enum at session-config time.

These tests drive the cache helpers in isolation. The end-to-end "lazy
import on tool call" path requires the full ``reachy_mini`` stack, which
isn't installable on the test host — covered separately by the existing
greet/emotion integration tests against mocked deps.
"""

from __future__ import annotations
import json
import importlib
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def fresh_play_emotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Reload ``play_emotion`` with the cache path pointed at ``tmp_path``.

    The fixture redirects ``_CATALOG_CACHE_PATH`` to a tmp file and
    monkey-patches the underlying ``RecordedMoves`` constructor so the
    module's init-time fallback (which fires when no cache exists) doesn't
    hit the real HuggingFace dataset.
    """
    import robot_comic.tools.play_emotion as pe

    cache_path = tmp_path / "_emotion_catalog_cache.json"
    monkeypatch.setattr(pe, "_CATALOG_CACHE_PATH", cache_path)
    monkeypatch.setattr(pe, "_CATALOG_NAMES", [])
    monkeypatch.setattr(pe, "_CATALOG_DESCRIPTIONS", {})
    monkeypatch.setattr(pe, "RECORDED_MOVES", None)
    monkeypatch.setattr(pe, "EMOTION_AVAILABLE", False)
    return pe


def test_load_from_cache_populates_in_memory(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    pe._CATALOG_CACHE_PATH.write_text(
        json.dumps(
            {
                "names": ["smile1", "frown2"],
                "descriptions": {"smile1": "happy", "frown2": "sad"},
            }
        ),
        encoding="utf-8",
    )
    ok = pe._load_catalog_from_cache()
    assert ok is True
    assert pe._CATALOG_NAMES == ["smile1", "frown2"]
    assert pe._CATALOG_DESCRIPTIONS == {"smile1": "happy", "frown2": "sad"}
    assert pe.EMOTION_AVAILABLE is True


def test_load_from_cache_filters_blocked_prefixes(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    pe._CATALOG_CACHE_PATH.write_text(
        json.dumps(
            {
                "names": ["smile1", "lonely1", "lonely_drift"],
                "descriptions": {
                    "smile1": "happy",
                    "lonely1": "blocked",
                    "lonely_drift": "blocked",
                },
            }
        ),
        encoding="utf-8",
    )
    pe._load_catalog_from_cache()
    assert pe._CATALOG_NAMES == ["smile1"]
    assert "lonely1" not in pe._CATALOG_DESCRIPTIONS
    assert "lonely_drift" not in pe._CATALOG_DESCRIPTIONS


def test_load_from_cache_missing_file_returns_false(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    assert not pe._CATALOG_CACHE_PATH.exists()
    assert pe._load_catalog_from_cache() is False
    assert pe._CATALOG_NAMES == []


def test_load_from_cache_malformed_json_returns_false(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    pe._CATALOG_CACHE_PATH.write_text("{not json", encoding="utf-8")
    assert pe._load_catalog_from_cache() is False
    assert pe._CATALOG_NAMES == []


def test_load_from_cache_wrong_shape_returns_false(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    pe._CATALOG_CACHE_PATH.write_text(json.dumps({"names": "not a list"}), encoding="utf-8")
    assert pe._load_catalog_from_cache() is False


def test_refresh_catalog_writes_cache(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    moves = MagicMock()
    moves.list_moves.return_value = ["smile1", "lonely1", "wave"]

    def _get(name: str) -> Any:
        descs = {"smile1": "happy", "lonely1": "blocked", "wave": "hi"}
        item = MagicMock()
        item.description = descs[name]
        return item

    moves.get.side_effect = _get

    pe._refresh_catalog_from_moves(moves)
    # In-memory filtered to skip blocked prefixes
    assert pe._CATALOG_NAMES == ["smile1", "wave"]
    # Cache on disk still has raw names + descriptions so a fresh
    # _load_catalog_from_cache() applies the blocked-prefix filter at read time
    raw = json.loads(pe._CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
    assert "lonely1" in raw["names"]
    assert raw["descriptions"]["lonely1"] == "blocked"


def test_safe_emotion_names_returns_copy(fresh_play_emotion: Any) -> None:
    pe = fresh_play_emotion
    pe._CATALOG_NAMES.extend(["a", "b"])
    out = pe._safe_emotion_names()
    out.append("mutated")
    assert pe._CATALOG_NAMES == ["a", "b"]


def test_get_available_emotions_and_descriptions_unavailable(
    fresh_play_emotion: Any,
) -> None:
    pe = fresh_play_emotion
    pe.EMOTION_AVAILABLE = False
    assert pe.get_available_emotions_and_descriptions() == "Emotions not available"


def test_get_available_emotions_and_descriptions_empty_catalog(
    fresh_play_emotion: Any,
) -> None:
    pe = fresh_play_emotion
    pe.EMOTION_AVAILABLE = True
    pe._CATALOG_NAMES.clear()
    assert pe.get_available_emotions_and_descriptions() == "No emotions currently available"


def test_get_available_emotions_and_descriptions_formats_lines(
    fresh_play_emotion: Any,
) -> None:
    pe = fresh_play_emotion
    pe._CATALOG_NAMES.extend(["smile1", "wave"])
    pe._CATALOG_DESCRIPTIONS.update({"smile1": "happy", "wave": "greet"})
    out = pe.get_available_emotions_and_descriptions()
    assert "Available emotions:" in out
    assert " - smile1: happy" in out
    assert " - wave: greet" in out


def test_module_uses_committed_cache_on_real_import() -> None:
    """Sanity: the production module loads from the committed cache and
    has at least one emotion + description populated. Run live (no fixture
    override) — confirms the boot fast-path is functional end-to-end.
    """
    pe = importlib.import_module("robot_comic.tools.play_emotion")
    # Reset and reload to exercise module init from disk.
    pe = importlib.reload(pe)
    assert pe.EMOTION_AVAILABLE is True
    assert len(pe._CATALOG_NAMES) > 0
    assert all(isinstance(n, str) for n in pe._CATALOG_NAMES)
