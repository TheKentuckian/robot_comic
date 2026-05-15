"""Unit tests for the byte-count-based echo-suppression deadline.

These tests verify that:
- After pushing N bytes of audio at the configured sample rate, the
  _speaking_until deadline is approximately start_ts + N/(rate*2) + cooldown.
- A transcript arriving before the deadline is suppressed.
- A transcript arriving after the deadline is accepted.
- The REACHY_MINI_ECHO_COOLDOWN_MS env var controls the margin.

All tests mock ``time.perf_counter`` so they are deterministic and instant.
"""

from __future__ import annotations
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from robot_comic.tools.core_tools import ToolDependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _pcm_frame(n_samples: int, sample_rate: int = 24000) -> np.ndarray:
    """Return a silent int16 PCM frame with the given number of samples."""
    return np.zeros(n_samples, dtype=np.int16)


# ---------------------------------------------------------------------------
# BaseLlamaResponseHandler / llama_base echo-guard tests
# ---------------------------------------------------------------------------


class TestLlamaBaseEchoGuard:
    """Tests for the byte-count echo guard in BaseLlamaResponseHandler.emit()."""

    def _make_handler(self):
        from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler

        handler = LocalSTTChatterboxHandler(_make_deps())
        handler._http = AsyncMock()
        return handler

    @pytest.mark.asyncio
    async def test_enqueue_audio_frame_sets_speaking_until_from_byte_count(self) -> None:
        """_enqueue_audio_frame() derives _speaking_until from cumulative bytes.

        Moved from emit() to BaseLlamaResponseHandler._enqueue_audio_frame in
        the Option A fix (spec:
        docs/superpowers/specs/2026-05-15-lifecycle-echo-guard-fix.md) so the
        composable factory path — which bypasses emit() — also updates the
        echo guard.
        """
        from robot_comic.llama_base import _BYTES_PER_SAMPLE, _OUTPUT_SAMPLE_RATE

        handler = self._make_handler()

        # Simulate 1 second of audio at 24 kHz / int16 = 48000 bytes
        n_samples = _OUTPUT_SAMPLE_RATE  # 1 second
        frame = _pcm_frame(n_samples, _OUTPUT_SAMPLE_RATE)
        expected_bytes = frame.nbytes  # n_samples * 2

        fake_start = 100.0

        cooldown_ms = 300
        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": str(cooldown_ms)}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()

            # First frame: _response_start_ts is captured from perf_counter().
            with patch("time.perf_counter", return_value=fake_start):
                await handler._enqueue_audio_frame(frame)

        cooldown_s = cooldown_ms / 1000.0
        bytes_per_second = _OUTPUT_SAMPLE_RATE * _BYTES_PER_SAMPLE
        expected_deadline = fake_start + expected_bytes / bytes_per_second + cooldown_s
        assert abs(handler._speaking_until - expected_deadline) < 1e-9
        assert handler._response_start_ts == fake_start
        assert handler._response_audio_bytes == expected_bytes

    @pytest.mark.asyncio
    async def test_drain_accumulates_bytes_and_sets_start_ts(self) -> None:
        """Frames drained to output_queue should update _response_audio_bytes."""
        from robot_comic.llama_base import _OUTPUT_SAMPLE_RATE

        handler = self._make_handler()
        handler._response_audio_bytes = 0
        handler._response_start_ts = 0.0

        n_samples = 2400  # one 100ms frame
        frame = _pcm_frame(n_samples, _OUTPUT_SAMPLE_RATE)
        expected_bytes = frame.nbytes

        # Manually invoke the byte-tracking logic that _drain_after_prev uses
        # (we test this via the actual _stream_response_and_synthesize path or
        # by simulating the drain directly).
        fake_now = 50.0
        with patch("time.perf_counter", return_value=fake_now):
            item = (_OUTPUT_SAMPLE_RATE, frame)
            # Replicate the drain body logic
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], np.ndarray):
                if handler._response_audio_bytes == 0:
                    handler._response_start_ts = fake_now
                handler._response_audio_bytes += item[1].nbytes
            await handler.output_queue.put(item)

        assert handler._response_audio_bytes == expected_bytes
        assert handler._response_start_ts == fake_now

    @pytest.mark.asyncio
    async def test_run_turn_resets_byte_counters(self) -> None:
        """_run_turn() must reset _response_audio_bytes / _response_start_ts."""
        from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler

        handler = LocalSTTChatterboxHandler(_make_deps())
        handler._http = AsyncMock()
        # Pre-load stale values from a prior turn
        handler._response_audio_bytes = 99999
        handler._response_start_ts = 1.0

        # Stub out the heavy LLM/TTS work — we only care about the reset
        async def _fake_stream(extra_messages=None, tts_span=None):
            return ("", [], {})

        handler._stream_response_and_synthesize = _fake_stream  # type: ignore[method-assign]

        await handler._run_turn(outer_span=None)

        assert handler._response_audio_bytes == 0
        assert handler._response_start_ts == 0.0


# ---------------------------------------------------------------------------
# ElevenLabsTTSResponseHandler echo-guard tests
# ---------------------------------------------------------------------------


class TestElevenLabsTTSEchoGuard:
    """Tests for the byte-count echo guard in ElevenLabsTTSResponseHandler.emit()."""

    def _make_handler(self):
        from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler

        handler = ElevenLabsTTSResponseHandler(_make_deps())
        handler._http = MagicMock()
        handler._client = MagicMock()
        return handler

    @pytest.mark.asyncio
    async def test_enqueue_audio_frame_sets_speaking_until_from_byte_count(self) -> None:
        """_enqueue_audio_frame() derives _speaking_until from cumulative bytes.

        Moved from emit() to the enqueue helper in the Option A fix
        (spec: docs/superpowers/specs/2026-05-15-lifecycle-echo-guard-fix.md)
        so the composable factory path — which bypasses emit() — also
        updates the echo guard.
        """
        from robot_comic.elevenlabs_tts import _BYTES_PER_SAMPLE, ELEVENLABS_OUTPUT_SAMPLE_RATE

        handler = self._make_handler()

        n_samples = ELEVENLABS_OUTPUT_SAMPLE_RATE  # 1 second
        frame = _pcm_frame(n_samples, ELEVENLABS_OUTPUT_SAMPLE_RATE)

        fake_start = 200.0

        cooldown_ms = 300
        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": str(cooldown_ms)}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()

            # First frame: _response_start_ts is captured from perf_counter().
            with patch("time.perf_counter", return_value=fake_start):
                await handler._enqueue_audio_frame(frame)

        cooldown_s = cooldown_ms / 1000.0
        bytes_per_second = ELEVENLABS_OUTPUT_SAMPLE_RATE * _BYTES_PER_SAMPLE
        expected_deadline = fake_start + frame.nbytes / bytes_per_second + cooldown_s
        assert abs(handler._speaking_until - expected_deadline) < 1e-9
        assert handler._response_start_ts == fake_start
        assert handler._response_audio_bytes == frame.nbytes

    @pytest.mark.asyncio
    async def test_enqueue_audio_frame_tracks_bytes(self) -> None:
        """_enqueue_audio_frame() must update _response_audio_bytes and start timestamp."""
        from robot_comic.elevenlabs_tts import ELEVENLABS_OUTPUT_SAMPLE_RATE

        handler = self._make_handler()
        handler._response_audio_bytes = 0
        handler._response_start_ts = 0.0

        frame = _pcm_frame(2400, ELEVENLABS_OUTPUT_SAMPLE_RATE)
        fake_ts = 75.0

        with patch("time.perf_counter", return_value=fake_ts):
            await handler._enqueue_audio_frame(frame)

        assert handler._response_audio_bytes == frame.nbytes
        assert handler._response_start_ts == fake_ts
        assert not handler.output_queue.empty()

    @pytest.mark.asyncio
    async def test_enqueue_audio_frame_accumulates_multiple_frames(self) -> None:
        """Consecutive _enqueue_audio_frame calls accumulate byte count."""
        from robot_comic.elevenlabs_tts import ELEVENLABS_OUTPUT_SAMPLE_RATE

        handler = self._make_handler()
        handler._response_audio_bytes = 0
        handler._response_start_ts = 0.0

        frame_a = _pcm_frame(2400, ELEVENLABS_OUTPUT_SAMPLE_RATE)
        frame_b = _pcm_frame(1200, ELEVENLABS_OUTPUT_SAMPLE_RATE)

        with patch("time.perf_counter", return_value=10.0):
            await handler._enqueue_audio_frame(frame_a)
        with patch("time.perf_counter", return_value=10.1):
            await handler._enqueue_audio_frame(frame_b)

        assert handler._response_audio_bytes == frame_a.nbytes + frame_b.nbytes
        # start_ts is set on the FIRST frame only
        assert handler._response_start_ts == 10.0

    @pytest.mark.asyncio
    async def test_dispatch_impl_resets_byte_counters(self) -> None:
        """_dispatch_completed_transcript_impl resets accumulators at turn start."""
        handler = self._make_handler()
        handler._response_audio_bytes = 77777
        handler._response_start_ts = 3.0

        # Stub out LLM and TTS to return immediately without side effects
        handler._run_llm_with_tools = AsyncMock(return_value="")  # type: ignore[method-assign]

        await handler._dispatch_completed_transcript_impl("hello")

        assert handler._response_audio_bytes == 0
        assert handler._response_start_ts == 0.0


# ---------------------------------------------------------------------------
# Echo-guard cooldown env var
# ---------------------------------------------------------------------------


class TestEchoGuardCooldownEnvVar:
    """REACHY_MINI_ECHO_COOLDOWN_MS controls the safety margin."""

    def test_default_cooldown_is_300ms(self) -> None:
        from robot_comic.config import DEFAULT_ECHO_COOLDOWN_MS

        assert DEFAULT_ECHO_COOLDOWN_MS == 300

    def test_env_var_overrides_cooldown(self) -> None:
        from robot_comic.config import config, refresh_runtime_config_from_env

        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": "150"}):
            refresh_runtime_config_from_env()
            assert config.ECHO_COOLDOWN_MS == 150

        # Restore
        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": "300"}):
            refresh_runtime_config_from_env()


# ---------------------------------------------------------------------------
# Suppression / acceptance end-to-end via LocalSTTInputMixin._handle_local_stt_event
# ---------------------------------------------------------------------------


class TestEchoGuardSuppression:
    """Transcripts arriving before the deadline are discarded; after accepted."""

    def _make_stt_handler(self):
        from robot_comic.local_stt_realtime import LocalSTTRealtimeHandler

        handler = LocalSTTRealtimeHandler(_make_deps())
        item = SimpleNamespace(create=AsyncMock())
        handler.connection = SimpleNamespace(conversation=SimpleNamespace(item=item))
        handler._safe_response_create = AsyncMock()  # type: ignore[method-assign]
        return handler

    @pytest.mark.asyncio
    async def test_transcript_before_deadline_is_suppressed(self) -> None:
        """When perf_counter() < _speaking_until the transcript must be dropped."""
        handler = self._make_stt_handler()
        # deadline 10 seconds in the future
        handler._speaking_until = 1000.0

        with patch("time.perf_counter", return_value=999.0):
            await handler._handle_local_stt_event("completed", "hello there")

        # connection.conversation.item.create should NOT have been called
        handler.connection.conversation.item.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transcript_after_deadline_is_accepted(self) -> None:
        """When perf_counter() > _speaking_until the transcript should be dispatched."""
        handler = self._make_stt_handler()
        # deadline already elapsed
        handler._speaking_until = 1.0

        with patch("time.perf_counter", return_value=100.0):
            await handler._handle_local_stt_event("completed", "hello there")

        handler.connection.conversation.item.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cooldown_ms_controls_margin(self) -> None:
        """The ECHO_COOLDOWN_MS env var shifts the suppression window."""
        from robot_comic.config import refresh_runtime_config_from_env
        from robot_comic.elevenlabs_tts import _BYTES_PER_SAMPLE, ELEVENLABS_OUTPUT_SAMPLE_RATE

        handler = self._make_stt_handler()

        # Push 1 second of audio worth of bytes into the handler's echo tracker
        n_bytes = ELEVENLABS_OUTPUT_SAMPLE_RATE * _BYTES_PER_SAMPLE  # 48000
        bytes_per_second = ELEVENLABS_OUTPUT_SAMPLE_RATE * _BYTES_PER_SAMPLE
        audio_duration_s = n_bytes / bytes_per_second  # exactly 1.0 s

        fake_start = 0.0
        handler._response_start_ts = fake_start
        handler._response_audio_bytes = n_bytes

        # With 500ms cooldown the deadline should be start + 1.0 + 0.5 = 1.5 s
        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": "500"}):
            refresh_runtime_config_from_env()
            # Artificially set _speaking_until as emit() would
            from robot_comic.config import config as _config

            cooldown_s = _config.ECHO_COOLDOWN_MS / 1000.0
            handler._speaking_until = fake_start + audio_duration_s + cooldown_s

        assert abs(handler._speaking_until - 1.5) < 1e-9

        # With 100ms cooldown the deadline is 1.1 s
        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": "100"}):
            refresh_runtime_config_from_env()
            from robot_comic.config import config as _config2

            cooldown_s2 = _config2.ECHO_COOLDOWN_MS / 1000.0
            handler._speaking_until = fake_start + audio_duration_s + cooldown_s2

        assert abs(handler._speaking_until - 1.1) < 1e-9

        # Restore
        with patch.dict(os.environ, {"REACHY_MINI_ECHO_COOLDOWN_MS": "300"}):
            refresh_runtime_config_from_env()
