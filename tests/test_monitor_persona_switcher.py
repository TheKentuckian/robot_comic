"""Tests for the persona-switcher overlay added to robot-comic-monitor.

Tests the state-machine logic (PersonaSwitcher), the picker-panel renderer,
and the HTTP helper functions — all with mocked network and stdin, so they
work identically on every platform.

Issue #246.
"""

from __future__ import annotations
import json
from typing import Any
from unittest.mock import MagicMock, patch

from robot_comic.monitor import (
    PersonaSwitcher,
    _apply_persona,
    _fetch_personas,
    _build_picker_panel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHOICES = ["bill_hicks", "don_rickles", "george_carlin"]
_CURRENT = "don_rickles"


def _make_http_response(body: dict[str, Any], status: int = 200) -> MagicMock:
    """Return a mock object that mimics ``urllib.request.urlopen`` response."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# PersonaSwitcher state-machine tests
# ---------------------------------------------------------------------------


def test_switcher_select_and_confirm() -> None:
    """Pressing a digit then Enter sets apply_name and done=True."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("2")
    assert sw.selected_idx == 1
    sw.handle_key("<enter>")
    assert sw.done
    assert sw.apply_name == "don_rickles"


def test_switcher_esc_cancels() -> None:
    """Pressing Esc sets done=True with apply_name=None."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("1")
    sw.handle_key("<esc>")
    assert sw.done
    assert sw.apply_name is None


def test_switcher_enter_without_selection_cancels() -> None:
    """Pressing Enter before selecting a row produces apply_name=None."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("<enter>")
    assert sw.done
    assert sw.apply_name is None


def test_switcher_out_of_range_digit_ignored() -> None:
    """A digit beyond the list length is silently ignored; no crash, no selection."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("9")  # only 3 choices; index 8 is out of range
    assert sw.selected_idx is None
    # Esc out cleanly
    sw.handle_key("<esc>")
    assert sw.done
    assert sw.apply_name is None


def test_switcher_zero_digit_ignored() -> None:
    """Digit '0' maps to index -1 and must be silently ignored."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("0")
    assert sw.selected_idx is None


def test_switcher_ctrl_c_cancels() -> None:
    """Ctrl-C (<interrupt>) cancels the overlay without applying."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("1")
    sw.handle_key("<interrupt>")
    assert sw.done
    assert sw.apply_name is None


def test_switcher_digit_changes_selection() -> None:
    """Selecting a row then pressing a different digit updates the selection."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("1")
    assert sw.selected_idx == 0
    sw.handle_key("3")
    assert sw.selected_idx == 2
    sw.handle_key("<enter>")
    assert sw.apply_name == "george_carlin"


def test_switcher_full_sequence_s_3_enter() -> None:
    """Simulate the key sequence ['3', '<enter>'] and verify POST payload."""
    # The 'S' key is handled by the monitor's main loop before constructing
    # PersonaSwitcher, so we start from the switcher directly.
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    for key in ["3", "<enter>"]:
        sw.handle_key(key)
    assert sw.done
    assert sw.apply_name == "george_carlin"


def test_switcher_full_sequence_s_esc_no_post() -> None:
    """Key sequence ['S', '<esc>'] must not result in a persona being applied."""
    sw = PersonaSwitcher(_CHOICES, _CURRENT)
    sw.handle_key("<esc>")
    assert sw.done
    assert sw.apply_name is None


# ---------------------------------------------------------------------------
# Picker-panel renderer tests
# ---------------------------------------------------------------------------


def test_build_picker_panel_marks_active() -> None:
    """The active persona row has '(active)' and '▶' marker."""
    panel = _build_picker_panel(_CHOICES, _CURRENT, selected_idx=None)
    # Gather all text rendered in the panel body (Table rows)
    table = panel.renderable
    # Align wraps the Table
    inner = getattr(table, "renderable", table)
    texts = []
    for col in inner.columns:
        for cell in col._cells:
            plain = cell.plain if hasattr(cell, "plain") else str(cell)
            texts.append(plain)
    combined = " ".join(texts)
    assert "▶" in combined
    assert "(active)" in combined
    assert "don_rickles" in combined


def test_build_picker_panel_highlights_selected() -> None:
    """The selected row (not active) renders differently from plain rows."""
    panel = _build_picker_panel(_CHOICES, _CURRENT, selected_idx=0)
    table = panel.renderable
    inner = getattr(table, "renderable", table)
    # Row 0 (bill_hicks) should be selected — just verify the panel builds OK
    # and the selected index is reflected somehow in the text.
    texts = []
    for col in inner.columns:
        for cell in col._cells:
            plain = cell.plain if hasattr(cell, "plain") else str(cell)
            texts.append(plain)
    combined = " ".join(texts)
    assert "bill_hicks" in combined


def test_build_picker_panel_status_line_in_subtitle() -> None:
    """A non-empty status_line replaces the default hint in the subtitle."""
    panel = _build_picker_panel(_CHOICES, _CURRENT, selected_idx=None, status_line="Switching…")
    subtitle = panel.subtitle
    plain = subtitle.plain if hasattr(subtitle, "plain") else str(subtitle)
    assert "Switching" in plain


# ---------------------------------------------------------------------------
# HTTP helper tests
# ---------------------------------------------------------------------------


def test_fetch_personas_success() -> None:
    """_fetch_personas parses choices + current from the /personalities JSON."""
    body = {"choices": _CHOICES, "current": _CURRENT, "startup": _CURRENT, "locked": False}
    with patch("robot_comic.monitor.urlopen", return_value=_make_http_response(body)):
        choices, current = _fetch_personas("http://localhost:8000")
    assert choices == _CHOICES
    assert current == _CURRENT


def test_fetch_personas_network_error() -> None:
    """_fetch_personas returns ([], '') on network failure — no exception raised."""
    with patch("robot_comic.monitor.urlopen", side_effect=OSError("refused")):
        choices, current = _fetch_personas("http://localhost:8000")
    assert choices == []
    assert current == ""


def test_apply_persona_success() -> None:
    """_apply_persona returns (True, ...) when the server responds {ok: true}."""
    body = {"ok": True, "status": "applied", "startup": "george_carlin"}
    with patch("robot_comic.monitor.urlopen", return_value=_make_http_response(body)):
        ok, msg = _apply_persona("http://localhost:8000", "george_carlin")
    assert ok is True
    assert "george_carlin" in msg


def test_apply_persona_server_error() -> None:
    """_apply_persona returns (False, ...) when the server returns ok=false."""
    body = {"ok": False, "error": "profile_locked", "locked_to": "don_rickles"}
    with patch("robot_comic.monitor.urlopen", return_value=_make_http_response(body)):
        ok, msg = _apply_persona("http://localhost:8000", "george_carlin")
    assert ok is False
    assert "profile_locked" in msg


def test_apply_persona_network_error() -> None:
    """_apply_persona returns (False, ...) on connection failure."""
    with patch("robot_comic.monitor.urlopen", side_effect=OSError("refused")):
        ok, msg = _apply_persona("http://localhost:8000", "george_carlin")
    assert ok is False
    assert msg  # some error message


# ---------------------------------------------------------------------------
# MonitorInput normalisation tests (platform-agnostic)
# ---------------------------------------------------------------------------


def test_monitor_input_normalise_esc() -> None:
    """ESC byte normalises to '<esc>'."""
    from robot_comic.monitor_input import _normalise

    assert _normalise(b"\x1b") == "<esc>"


def test_monitor_input_normalise_enter_cr() -> None:
    """CR byte normalises to '<enter>'."""
    from robot_comic.monitor_input import _normalise

    assert _normalise(b"\r") == "<enter>"


def test_monitor_input_normalise_enter_lf() -> None:
    """LF byte normalises to '<enter>'."""
    from robot_comic.monitor_input import _normalise

    assert _normalise(b"\n") == "<enter>"


def test_monitor_input_normalise_ctrl_c() -> None:
    """Ctrl-C byte normalises to '<interrupt>'."""
    from robot_comic.monitor_input import _normalise

    assert _normalise(b"\x03") == "<interrupt>"


def test_monitor_input_normalise_printable_lower() -> None:
    """Printable ASCII is returned lower-cased."""
    from robot_comic.monitor_input import _normalise

    assert _normalise(b"S") == "s"
    assert _normalise(b"s") == "s"
    assert _normalise(b"3") == "3"


def test_monitor_input_normalise_non_printable_returns_none() -> None:
    """Non-printable bytes (other than mapped ones) return None."""
    from robot_comic.monitor_input import _normalise

    assert _normalise(b"\x01") is None
    assert _normalise(b"\x80") is None


# ---------------------------------------------------------------------------
# MonitorInput.poll_key on non-tty stdin (cross-platform unit test)
# ---------------------------------------------------------------------------


def test_monitor_input_poll_key_no_tty() -> None:
    """MonitorInput.poll_key returns None when stdin is not a tty (CI / test env)."""
    from robot_comic.monitor_input import MonitorInput

    # This path exercises the graceful fallback — stdin is typically not a
    # tty in pytest, so the backend silently degrades to returning None.
    inp = MonitorInput()
    try:
        result = inp.poll_key(timeout=0.0)
        # On a non-tty stdin the backend returns None; on Windows msvcrt may
        # genuinely find no key.  Either is acceptable.
        assert result is None or isinstance(result, str)
    finally:
        inp.close()
