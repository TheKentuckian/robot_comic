"""Unit tests for audio_gain.normalize_gain and its integration in ChatterboxTTS.

All tests are pure numerical — no audio files, no network connections, no
Chatterbox server required.
"""

from __future__ import annotations
import io
import math
import wave

import numpy as np
import pytest

from robot_comic.audio_gain import _SILENCE_FLOOR_RMS, normalize_gain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm(values: list[int]) -> "np.ndarray[np.int16]":
    """Build an int16 ndarray from a list of sample values."""
    return np.array(values, dtype=np.int16)


def _rms_dbfs(pcm: "np.ndarray[np.int16]") -> float:
    """Return the RMS level of a PCM array in dBFS (relative to int16 full scale)."""
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    if rms == 0.0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)


def _sine_pcm(amplitude: int, n_samples: int = 4800) -> "np.ndarray[np.int16]":
    """Generate a mono sine-wave PCM array at a given amplitude (int16 peak)."""
    t = np.arange(n_samples, dtype=np.float32) / n_samples
    audio = (np.sin(2 * math.pi * 10 * t) * amplitude).astype(np.int16)
    return audio


def _make_wav_bytes(pcm: "np.ndarray[np.int16]", sample_rate: int = 24000) -> bytes:
    """Wrap a mono int16 PCM array into raw WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# normalize_gain unit tests
# ---------------------------------------------------------------------------


class TestNormalizeGain:
    """Direct tests for the normalize_gain helper in audio_gain.py."""

    def test_silent_input_unchanged(self) -> None:
        """Silent PCM (all zeros) must be returned unchanged."""
        pcm = _make_pcm([0] * 1000)
        result = normalize_gain(pcm, target_dbfs=-16.0)
        np.testing.assert_array_equal(result, pcm)

    def test_empty_input_unchanged(self) -> None:
        """Empty PCM array must be returned unchanged without error."""
        pcm = np.array([], dtype=np.int16)
        result = normalize_gain(pcm, target_dbfs=-16.0)
        assert result.size == 0

    def test_loud_tone_scaled_down(self) -> None:
        """A tone near full scale should be attenuated to the target level."""
        # ~−1 dBFS peak amplitude → RMS ≈ −4 dBFS
        pcm = _sine_pcm(amplitude=30000)
        target = -16.0
        result = normalize_gain(pcm, target_dbfs=target)
        actual_dbfs = _rms_dbfs(result)
        assert abs(actual_dbfs - target) < 1.0, (
            f"Expected ≈{target} dBFS after normalization, got {actual_dbfs:.2f} dBFS"
        )

    def test_quiet_tone_scaled_up(self) -> None:
        """A very quiet tone should be amplified to the target level."""
        # Amplitude of 300 → RMS ≈ −40 dBFS
        pcm = _sine_pcm(amplitude=300)
        target = -16.0
        result = normalize_gain(pcm, target_dbfs=target)
        actual_dbfs = _rms_dbfs(result)
        assert abs(actual_dbfs - target) < 1.0, (
            f"Expected ≈{target} dBFS after normalization, got {actual_dbfs:.2f} dBFS"
        )

    def test_output_never_exceeds_int16_max(self) -> None:
        """Output samples must be clipped to the int16 range regardless of gain."""
        # Use a target that would try to boost a loud tone above clipping
        pcm = _sine_pcm(amplitude=32000)
        # Trying to bring a near-full-scale tone TO 0 dBFS would clip;
        # verify the clipping guard at least keeps values in range.
        result = normalize_gain(pcm, target_dbfs=0.0)
        assert int(result.max()) <= 32767
        assert int(result.min()) >= -32768

    def test_output_dtype_is_int16(self) -> None:
        """normalize_gain must always return an int16 array."""
        pcm = _sine_pcm(amplitude=10000)
        result = normalize_gain(pcm, target_dbfs=-16.0)
        assert result.dtype == np.int16

    def test_below_silence_floor_not_amplified(self) -> None:
        """PCM whose RMS is below the silence floor must be returned as-is."""
        # Build a 1-sample signal right at the silence floor boundary (just below).
        threshold_count = int(_SILENCE_FLOOR_RMS * 0.5)
        pcm = _make_pcm([threshold_count] * 100)
        result = normalize_gain(pcm, target_dbfs=-16.0)
        np.testing.assert_array_equal(result, pcm)

    def test_warning_logged_on_clipping(self, caplog: pytest.LogCaptureFixture) -> None:
        """A WARNING must be logged when hard clipping is triggered."""
        import logging

        # Very loud source + aggressive (too high) target → clipping
        pcm = _sine_pcm(amplitude=32000)
        with caplog.at_level(logging.WARNING, logger="robot_comic.audio_gain"):
            normalize_gain(pcm, target_dbfs=0.0)
        assert any("hard clipping" in record.message for record in caplog.records)

    def test_no_warning_when_no_clipping(self, caplog: pytest.LogCaptureFixture) -> None:
        """No WARNING should be emitted when clipping does not occur."""
        import logging

        pcm = _sine_pcm(amplitude=10000)
        with caplog.at_level(logging.WARNING, logger="robot_comic.audio_gain"):
            normalize_gain(pcm, target_dbfs=-16.0)
        assert not any("hard clipping" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Integration: ChatterboxTTSResponseHandler._wav_to_pcm
# ---------------------------------------------------------------------------


class TestWavToPcmAutoGain:
    """Integration tests verifying auto-gain inside _wav_to_pcm."""

    def _call_wav_to_pcm(
        self,
        pcm_in: "np.ndarray[np.int16]",
        gain: float = 1.0,
        auto_gain: bool = True,
        target_dbfs: float = -16.0,
    ) -> "np.ndarray[np.int16]":
        from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

        wav_bytes = _make_wav_bytes(pcm_in, sample_rate=24000)
        raw = ChatterboxTTSResponseHandler._wav_to_pcm(
            wav_bytes, gain=gain, auto_gain=auto_gain, target_dbfs=target_dbfs
        )
        return np.frombuffer(raw, dtype=np.int16)

    def test_auto_gain_disabled_does_not_normalize(self) -> None:
        """When auto_gain=False the output RMS should not match target_dbfs."""
        loud_pcm = _sine_pcm(amplitude=30000)
        target = -16.0
        result = self._call_wav_to_pcm(loud_pcm, auto_gain=False, target_dbfs=target)
        actual_dbfs = _rms_dbfs(result)
        # The source is already at ≈ −4 dBFS — normalization is NOT applied.
        # We just verify it is NOT pulled exactly to −16 dBFS.
        assert abs(actual_dbfs - target) > 1.0, "Expected normalization to be skipped when auto_gain=False"

    def test_auto_gain_enabled_reaches_target(self) -> None:
        """When auto_gain=True the output should be within 1 dB of target_dbfs."""
        quiet_pcm = _sine_pcm(amplitude=1000)
        target = -16.0
        result = self._call_wav_to_pcm(quiet_pcm, auto_gain=True, target_dbfs=target)
        actual_dbfs = _rms_dbfs(result)
        assert abs(actual_dbfs - target) < 1.0, f"Expected ≈{target} dBFS, got {actual_dbfs:.2f} dBFS"

    def test_output_clipped_to_int16(self) -> None:
        """_wav_to_pcm output must always be within int16 bounds."""
        loud_pcm = _sine_pcm(amplitude=32000)
        result = self._call_wav_to_pcm(loud_pcm, auto_gain=True, target_dbfs=0.0)
        assert int(result.max()) <= 32767
        assert int(result.min()) >= -32768
