"""Tests for joke_history module."""

from __future__ import annotations
import os
import json
from pathlib import Path
from unittest.mock import patch

from robot_comic.joke_history import JokeHistory, last_sentence_of, default_history_path


# ---------------------------------------------------------------------------
# last_sentence_of
# ---------------------------------------------------------------------------


def test_last_sentence_single_terminator() -> None:
    assert last_sentence_of("Hello. World!") == "World!"


def test_last_sentence_no_terminator() -> None:
    assert last_sentence_of("Just one") == "Just one"


def test_last_sentence_empty() -> None:
    assert last_sentence_of("") == ""


def test_last_sentence_whitespace_only() -> None:
    assert last_sentence_of("   ") == ""


def test_last_sentence_multiple_terminators() -> None:
    result = last_sentence_of("First. Second? Third!")
    assert result == "Third!"


def test_last_sentence_strips_leading_whitespace() -> None:
    assert last_sentence_of("  Hello.  World!  ") == "World!"


# ---------------------------------------------------------------------------
# JokeHistory.load
# ---------------------------------------------------------------------------


def test_load_missing_file(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "nonexistent.json")
    assert history.load() == []


def test_load_empty_entries(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("[]", encoding="utf-8")
    history = JokeHistory(path)
    assert history._entries == []


def test_load_existing_entries(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    entries = [{"ts": "2024-01-01T00:00:00+00:00", "punchline": "Nice tie!", "topic": "fashion"}]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    assert len(history._entries) == 1
    assert history._entries[0]["punchline"] == "Nice tie!"


def test_load_corrupt_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("not json{{{{", encoding="utf-8")
    history = JokeHistory(path)
    assert history._entries == []


def test_load_wrong_type_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    history = JokeHistory(path)
    assert history._entries == []


# ---------------------------------------------------------------------------
# JokeHistory.add / truncation
# ---------------------------------------------------------------------------


def test_add_appends_entry(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json", max_entries=50)
    history.add("You're so ugly, mirrors refuse to look at you!")
    assert len(history._entries) == 1
    assert history._entries[0]["punchline"] == "You're so ugly, mirrors refuse to look at you!"
    assert history._entries[0]["topic"] == ""


def test_add_with_topic(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("Great haircut!", topic="appearance")
    assert history._entries[0]["topic"] == "appearance"


def test_add_truncates_beyond_max(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json", max_entries=3)
    for i in range(5):
        history.add(f"joke {i}")
    assert len(history._entries) == 3
    # Only the last 3 should remain
    punchlines = [e["punchline"] for e in history._entries]
    assert punchlines == ["joke 2", "joke 3", "joke 4"]


def test_add_empty_punchline_skipped(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("")
    history.add("   ")
    assert history._entries == []


def test_add_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = JokeHistory(path)
    history.add("Saved joke!")
    # Read back raw
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["punchline"] == "Saved joke!"


# ---------------------------------------------------------------------------
# JokeHistory.save (atomic write)
# ---------------------------------------------------------------------------


def test_save_writes_correct_content(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = JokeHistory(path)
    entries = [{"ts": "2024-01-01T00:00:00+00:00", "punchline": "atomic!", "topic": ""}]
    history.save(entries)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == entries


def test_save_is_atomic(tmp_path: Path) -> None:
    """Verify atomic write: replace is called, not a direct open-write."""
    path = tmp_path / "history.json"
    history = JokeHistory(path)
    replace_calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def _mock_replace(src: str, dst: str) -> None:
        replace_calls.append((src, dst))
        original_replace(src, dst)

    with patch("os.replace", side_effect=_mock_replace):
        history.save([{"ts": "x", "punchline": "test", "topic": ""}])

    assert len(replace_calls) == 1
    _, dst = replace_calls[0]
    assert Path(dst) == path


# ---------------------------------------------------------------------------
# JokeHistory.recent
# ---------------------------------------------------------------------------


def test_recent_returns_last_n(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    for i in range(15):
        history.add(f"joke {i}")
    recent = history.recent(n=5)
    assert len(recent) == 5
    punchlines = [e["punchline"] for e in recent]
    assert punchlines == ["joke 10", "joke 11", "joke 12", "joke 13", "joke 14"]


def test_recent_returns_all_when_fewer_than_n(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("only one")
    recent = history.recent(n=10)
    assert len(recent) == 1


def test_recent_empty_history(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    assert history.recent() == []


def test_recent_preserves_order(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("first")
    history.add("second")
    history.add("third")
    recent = history.recent(n=3)
    punchlines = [e["punchline"] for e in recent]
    assert punchlines == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# JokeHistory.format_for_prompt
# ---------------------------------------------------------------------------


def test_format_for_prompt_empty_returns_empty_string(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    assert history.format_for_prompt() == ""


def test_format_for_prompt_nonempty_contains_bullets(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("Nice tie!")
    history.add("Your hair is thinning faster than my patience.")
    result = history.format_for_prompt()
    assert result != ""
    assert "## RECENT JOKES (DO NOT REPEAT)" in result
    assert "- Nice tie!" in result
    assert "- Your hair is thinning faster than my patience." in result


def test_format_for_prompt_respects_n(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    for i in range(10):
        history.add(f"joke {i}")
    result = history.format_for_prompt(n=3)
    # Only the last 3 should appear
    assert "joke 7" in result
    assert "joke 8" in result
    assert "joke 9" in result
    assert "joke 0" not in result


def test_format_for_prompt_one_bullet_per_entry(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("alpha")
    history.add("beta")
    result = history.format_for_prompt()
    assert result.count("- alpha") == 1
    assert result.count("- beta") == 1


# ---------------------------------------------------------------------------
# default_history_path
# ---------------------------------------------------------------------------


def test_default_history_path_returns_path_under_home(tmp_path: Path) -> None:
    with patch("robot_comic.joke_history._DEFAULT_HISTORY_DIR", tmp_path / ".robot-comic"):
        path = default_history_path()
    assert path.name == "joke-history.json"
    assert (tmp_path / ".robot-comic").is_dir()
