"""Regression tests for the telemetry housekeeping + creds-guard fixes.

Covers the four spec items in
``docs/superpowers/specs/2026-05-16-telemetry-housekeeping-and-creds-guard.md``:

Fix 1 — Orphan OTel attribute allowlist exposure (telemetry._SPAN_ATTRS_TO_KEEP).
Fix 2 — ``telemetry.errors`` + ``telemetry.playback_underruns`` have call sites.
Fix 3 — ``_prepare_startup_credentials`` idempotency guard on the shared host.

(Fix 4 — ``handler.start_up.complete`` event timing — lives in
``tests/test_composable_conversation_handler.py``.)
"""

from __future__ import annotations
from typing import Any
from unittest.mock import patch

import pytest

from robot_comic import telemetry


# ---------------------------------------------------------------------------
# Fix 1 — allowlist exposes the four orphan attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr_name",
    [
        "gen_ai.usage.api_call_count",
        "gen_ai.server.time_to_first_token",
        "tts.voice_id",
        "tts.char_count",
    ],
)
def test_orphan_span_attributes_now_in_allowlist(attr_name: str) -> None:
    """Each of the four attributes flagged by instrumentation audit §3/§4 is
    actually set somewhere in ``src/`` (verified via grep at spec authoring
    time) but used to be dropped by ``CompactLineExporter`` because the
    allowlist didn't include them. Fix 1 adds them.
    """
    assert attr_name in telemetry._SPAN_ATTRS_TO_KEEP, (
        f"{attr_name!r} should be in telemetry._SPAN_ATTRS_TO_KEEP so the "
        "exporter doesn't drop it on its way to the monitor"
    )


# ---------------------------------------------------------------------------
# Fix 2a — ``telemetry.errors`` is wired at Moonshine warning sites
# ---------------------------------------------------------------------------


def test_inc_errors_called_from_local_stt_on_error_listener() -> None:
    """``_MoonshineListener.on_error`` should now increment the errors
    counter alongside the existing ``logger.warning``. Instrumentation
    audit Rec 3.
    """
    from robot_comic.local_stt_realtime import _MoonshineListener

    class _StubHandler:
        _pending_stream_rearm = False
        # _diag_enabled() reads env; we only need a stand-in attribute path.

    listener = object.__new__(_MoonshineListener)
    listener.handler = _StubHandler()  # type: ignore[attr-defined]

    class _Event:
        error = "stream borked"

    with patch.object(telemetry, "inc_errors") as mock_inc:
        listener.on_error(_Event())

    assert mock_inc.call_count == 1, (
        f"on_error must call telemetry.inc_errors once; got {mock_inc.call_count} calls: {mock_inc.call_args_list!r}"
    )
    (attrs,), _ = mock_inc.call_args
    assert attrs.get("subsystem") == "stt"
    assert "error_type" in attrs


# ---------------------------------------------------------------------------
# Fix 2b — ``telemetry.playback_underruns`` is wired at the welcome-WAV path
# ---------------------------------------------------------------------------


def test_inc_playback_underruns_fires_on_nonzero_aplay_exit() -> None:
    """A non-zero ``aplay`` exit on the welcome-WAV completion-now path is
    the one playback-underrun signal we control directly. Instrumentation
    audit Rec 7.
    """
    from robot_comic import warmup_audio

    class _ExitedPopen:
        returncode = 1

    with (
        patch.object(telemetry, "inc_playback_underruns") as mock_inc,
        patch.object(telemetry, "emit_supporting_event"),
    ):
        warmup_audio._emit_completion_now(
            _ExitedPopen(),  # type: ignore[arg-type]
            command=["aplay", "x.wav"],
            started_at=0.0,
        )

    assert mock_inc.call_count == 1, (
        f"Non-zero exit must increment playback_underruns; got {mock_inc.call_count} calls"
    )
    (attrs,), _ = mock_inc.call_args
    assert attrs.get("path") == "welcome.wav"


def test_inc_playback_underruns_skipped_on_clean_exit() -> None:
    """Zero exit code = playback completed cleanly. No underrun."""
    from robot_comic import warmup_audio

    class _CleanPopen:
        returncode = 0

    with (
        patch.object(telemetry, "inc_playback_underruns") as mock_inc,
        patch.object(telemetry, "emit_supporting_event"),
    ):
        warmup_audio._emit_completion_now(
            _CleanPopen(),  # type: ignore[arg-type]
            command=["aplay", "x.wav"],
            started_at=0.0,
        )

    assert mock_inc.call_count == 0, (
        f"Clean exit must not increment playback_underruns; got {mock_inc.call_count} calls"
    )


# ---------------------------------------------------------------------------
# Fix 3 — _prepare_startup_credentials idempotency guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_startup_credentials_only_runs_once_for_shared_host() -> None:
    """The three composable adapters (LLM, TTS, STT) all call
    ``_prepare_startup_credentials`` on the shared host instance during their
    ``prepare``/``start`` lifecycles. Without a guard the underlying work
    (httpx client, Gemini client, Moonshine model load) runs three times
    serially during the start_up critical path. The guard collapses it to
    one real call. Boot memo (PR #383) §V7.
    """
    from robot_comic.local_stt_realtime import LocalSTTInputMixin

    super_calls: list[None] = []

    class _CountingBase:
        async def _prepare_startup_credentials(self) -> None:
            super_calls.append(None)

    class _Host(LocalSTTInputMixin, _CountingBase):
        # Skip the heavy mixin __init__ — we only care about the method
        # under test. ``object.__new__`` + manual attribute set keeps the
        # test independent of LocalSTTInputMixin's wider state.
        def __init__(self) -> None:
            self._startup_credentials_ready = False
            self._local_loop = None

    host = _Host()
    host._build_local_stt_stream = lambda: None  # type: ignore[method-assign]

    # Simulate the three adapter prepare() calls happening sequentially
    # against the same host (the live behaviour today).
    await host._prepare_startup_credentials()
    await host._prepare_startup_credentials()
    await host._prepare_startup_credentials()

    assert len(super_calls) == 1, (
        "Idempotency guard failed: super()._prepare_startup_credentials was "
        f"called {len(super_calls)} times; expected exactly 1."
    )
    assert host._startup_credentials_ready is True


@pytest.mark.asyncio
async def test_prepare_startup_credentials_retries_after_failure() -> None:
    """The guard must NOT lock out retries when the first call raises —
    operators expect a transient failure (network, missing API key, etc.) to
    re-attempt on the next adapter's ``prepare()``.
    """
    from robot_comic.local_stt_realtime import LocalSTTInputMixin

    attempt = {"n": 0}

    class _FlakyBase:
        async def _prepare_startup_credentials(self) -> None:
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise RuntimeError("transient failure")

    class _Host(LocalSTTInputMixin, _FlakyBase):
        def __init__(self) -> None:
            self._startup_credentials_ready = False
            self._local_loop = None

    host = _Host()
    host._build_local_stt_stream = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="transient failure"):
        await host._prepare_startup_credentials()
    assert host._startup_credentials_ready is False, "Guard flipped after a failed call — would lock out retries"

    # Second attempt should succeed.
    await host._prepare_startup_credentials()
    assert attempt["n"] == 2
    assert host._startup_credentials_ready is True

    # Third attempt is a no-op.
    await host._prepare_startup_credentials()
    assert attempt["n"] == 2, "Guard did not short-circuit after first success"


# ---------------------------------------------------------------------------
# Smoke: the housekeeping helpers themselves still no-op cleanly when
# instrumentation is not initialised (no Meter wired).
# ---------------------------------------------------------------------------


def test_inc_errors_is_safe_when_uninitialised() -> None:
    """Both new call sites might fire before ``telemetry.init()`` (e.g. a
    Moonshine on_error during cold-boot warm-up). The counter helper must
    no-op silently in that state."""
    # Force the no-op path: the counter is unset until init() runs.
    saved = telemetry.errors
    telemetry.errors = None
    try:
        telemetry.inc_errors({"subsystem": "stt", "error_type": "x"})
    finally:
        telemetry.errors = saved


def test_inc_playback_underruns_is_safe_when_uninitialised() -> None:
    """Same contract as ``inc_errors``."""
    saved = telemetry.playback_underruns
    telemetry.playback_underruns = None
    try:
        telemetry.inc_playback_underruns({"path": "welcome.wav"})
    finally:
        telemetry.playback_underruns = saved


def test_inc_errors_arg_shape_matches_call_sites() -> None:
    """Documentation guard: the helper accepts dict-attributes (no positional
    count argument), matching every call site in src/."""
    import inspect

    sig = inspect.signature(telemetry.inc_errors)
    params = list(sig.parameters.values())
    # One required positional: ``attrs``.
    assert any(p.name == "attrs" for p in params), (
        f"telemetry.inc_errors signature changed; new call sites may be wrong: {sig}"
    )


def test_compact_line_exporter_keeps_new_attributes() -> None:
    """End-to-end-ish: build a fake span carrying the four newly-allowlisted
    attributes and confirm the exporter does not drop them. Belt-and-braces
    around the ``frozenset`` membership test above.
    """
    exporter = telemetry.CompactLineExporter()

    class _SpanCtx:
        trace_id = 0x1
        span_id = 0x2

    class _FakeSpan:
        name = "tts.synthesize"
        attributes: dict[str, Any] = {
            "gen_ai.usage.api_call_count": 3,
            "gen_ai.server.time_to_first_token": 0.42,
            "tts.voice_id": "rickles_ivc",
            "tts.char_count": 128,
            # An attribute that is NOT in the allowlist — must be filtered.
            "something.random": "drop me",
        }
        context = _SpanCtx()
        parent = None
        start_time = 0
        end_time = 1_000_000

        class status:
            class status_code:
                name = "OK"

    captured: list[str] = []

    def _capture(s: str, flush: bool = False) -> None:  # noqa: ARG001
        captured.append(s)

    with patch("builtins.print", side_effect=_capture):
        result = exporter.export([_FakeSpan()])

    assert result == telemetry.SpanExportResult.SUCCESS
    assert len(captured) == 1, captured
    line = captured[0]
    assert "gen_ai.usage.api_call_count" in line
    assert "gen_ai.server.time_to_first_token" in line
    assert "tts.voice_id" in line
    assert "tts.char_count" in line
    # And the non-allowlisted attr is dropped.
    assert "something.random" not in line
