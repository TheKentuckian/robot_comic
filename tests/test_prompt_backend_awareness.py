"""Unit tests for backend-aware GEMINI TTS DELIVERY TAGS section filtering.

Covers Issue #102 — the ``_filter_delivery_tags`` helper in ``prompts.py``
should include or strip the delivery-tags section depending on the active
audio output backend, and an env-var override must bypass the stripping.

Post-Phase-4f (PR retiring ``BACKEND_PROVIDER`` /
``LOCAL_STT_RESPONSE_BACKEND``) the gate reads ``config.AUDIO_OUTPUT_BACKEND``
directly: the section is kept only when the active audio output is
``AUDIO_OUTPUT_GEMINI_TTS``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from robot_comic.config import (
    AUDIO_OUTPUT_HF,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_OPENAI_REALTIME,
)
from robot_comic.prompts import (
    _uses_gemini_tts,
    _strip_tts_section,
    _filter_delivery_tags,
    get_session_instructions,
)


_SAMPLE_INSTRUCTIONS = """\
## IDENTITY

You are a test persona.

## PHYSICAL BEATS

Do stuff.

## GEMINI TTS DELIVERY TAGS

Use [fast] and [annoyance] freely.

**Rule:** one tag per sentence.

## GEMINI LIVE DELIVERY GUIDANCE

Perform live.
"""

_SECTION_HEADER = "## GEMINI TTS DELIVERY TAGS"


def _make_fake_config(
    *,
    audio_output_backend: str,
    force_delivery_tags: bool = False,
) -> object:
    """Return a minimal config-like namespace for patching."""

    class _FakeConfig:
        AUDIO_OUTPUT_BACKEND = audio_output_backend
        FORCE_DELIVERY_TAGS = force_delivery_tags

    return _FakeConfig()


# ---------------------------------------------------------------------------
# _uses_gemini_tts
# ---------------------------------------------------------------------------


def test_uses_gemini_tts_output_is_true() -> None:
    """gemini_tts output must return True."""
    assert _uses_gemini_tts(AUDIO_OUTPUT_GEMINI_TTS) is True


def test_uses_gemini_live_output_is_false() -> None:
    """Gemini Live output does NOT consume TTS delivery tags."""
    assert _uses_gemini_tts(AUDIO_OUTPUT_GEMINI_LIVE) is False


def test_uses_hf_output_is_false() -> None:
    """Hugging Face output does not consume TTS tags."""
    assert _uses_gemini_tts(AUDIO_OUTPUT_HF) is False


def test_uses_openai_realtime_output_is_false() -> None:
    """OpenAI Realtime output does not consume TTS tags."""
    assert _uses_gemini_tts(AUDIO_OUTPUT_OPENAI_REALTIME) is False


def test_uses_chatterbox_output_is_false() -> None:
    """Chatterbox output does not consume TTS tags."""
    assert _uses_gemini_tts(AUDIO_OUTPUT_CHATTERBOX) is False


def test_uses_elevenlabs_output_is_false() -> None:
    """ElevenLabs output does not consume TTS tags."""
    assert _uses_gemini_tts(AUDIO_OUTPUT_ELEVENLABS) is False


def test_uses_empty_string_is_false() -> None:
    """Defensive: an unconfigured AUDIO_OUTPUT_BACKEND must not keep the section."""
    assert _uses_gemini_tts("") is False


# ---------------------------------------------------------------------------
# _strip_tts_section
# ---------------------------------------------------------------------------


def test_strip_removes_section() -> None:
    """The GEMINI TTS DELIVERY TAGS section is removed."""
    result = _strip_tts_section(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_strip_preserves_content_after_section() -> None:
    """Content following the TTS section is kept."""
    result = _strip_tts_section(_SAMPLE_INSTRUCTIONS)
    assert "## GEMINI LIVE DELIVERY GUIDANCE" in result
    assert "Perform live." in result


def test_strip_preserves_content_before_section() -> None:
    """Content preceding the TTS section is kept."""
    result = _strip_tts_section(_SAMPLE_INSTRUCTIONS)
    assert "## IDENTITY" in result
    assert "## PHYSICAL BEATS" in result


def test_strip_no_triple_blank_lines() -> None:
    """Stripping does not leave triple-blank-line gaps."""
    result = _strip_tts_section(_SAMPLE_INSTRUCTIONS)
    assert "\n\n\n" not in result


def test_strip_no_section_is_idempotent() -> None:
    """Instructions without the section pass through unchanged."""
    plain = "## IDENTITY\n\nHello.\n"
    assert _strip_tts_section(plain).strip() == plain.strip()


# ---------------------------------------------------------------------------
# _filter_delivery_tags
# ---------------------------------------------------------------------------


def test_filter_gemini_tts_output_keeps_section() -> None:
    """gemini_tts output keeps the delivery-tags section."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_GEMINI_TTS)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_gemini_live_output_strips_section() -> None:
    """Gemini Live output strips the delivery-tags section."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_GEMINI_LIVE)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_hf_output_strips_section() -> None:
    """Hugging Face output strips the delivery-tags section."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_HF)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_openai_realtime_output_strips_section() -> None:
    """OpenAI Realtime output strips the delivery-tags section."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_OPENAI_REALTIME)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_chatterbox_output_strips_section() -> None:
    """Chatterbox output strips the delivery-tags section."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_CHATTERBOX)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_elevenlabs_output_strips_section() -> None:
    """ElevenLabs output strips the delivery-tags section."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_ELEVENLABS)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_force_override_keeps_section_on_chatterbox() -> None:
    """REACHY_MINI_FORCE_DELIVERY_TAGS=1 bypasses stripping on Chatterbox."""
    fake = _make_fake_config(
        audio_output_backend=AUDIO_OUTPUT_CHATTERBOX, force_delivery_tags=True
    )
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_force_override_keeps_section_on_hf() -> None:
    """REACHY_MINI_FORCE_DELIVERY_TAGS=1 bypasses stripping on HuggingFace."""
    fake = _make_fake_config(
        audio_output_backend=AUDIO_OUTPUT_HF, force_delivery_tags=True
    )
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_force_false_does_not_keep_section_on_chatterbox() -> None:
    """FORCE_DELIVERY_TAGS=False still strips on Chatterbox."""
    fake = _make_fake_config(
        audio_output_backend=AUDIO_OUTPUT_CHATTERBOX, force_delivery_tags=False
    )
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_stripped_result_preserves_live_guidance() -> None:
    """Stripping TTS section leaves GEMINI LIVE DELIVERY GUIDANCE intact."""
    fake = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_GEMINI_LIVE)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert "## GEMINI LIVE DELIVERY GUIDANCE" in result
    assert "Perform live." in result


# ---------------------------------------------------------------------------
# get_session_instructions integration
# ---------------------------------------------------------------------------


def _write_test_profile(tmp_path: Path) -> Path:
    """Write a test profile with GEMINI TTS DELIVERY TAGS section and return profiles root."""
    profile_dir = tmp_path / "test_persona"
    profile_dir.mkdir()
    (profile_dir / "instructions.txt").write_text(_SAMPLE_INSTRUCTIONS, encoding="utf-8")
    return tmp_path


def test_get_session_instructions_chatterbox_strips_section(tmp_path: Path) -> None:
    """get_session_instructions() strips the delivery section for Chatterbox."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_CHATTERBOX)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER not in result
    assert "## IDENTITY" in result


def test_get_session_instructions_gemini_tts_keeps_section(tmp_path: Path) -> None:
    """get_session_instructions() keeps the delivery section for Gemini TTS."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(audio_output_backend=AUDIO_OUTPUT_GEMINI_TTS)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER in result


def test_get_session_instructions_force_override_keeps_section_on_hf(tmp_path: Path) -> None:
    """get_session_instructions() respects FORCE_DELIVERY_TAGS on HuggingFace."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(
        audio_output_backend=AUDIO_OUTPUT_HF, force_delivery_tags=True
    )
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER in result


@pytest.mark.parametrize(
    "audio_output_backend",
    [
        AUDIO_OUTPUT_GEMINI_LIVE,
        AUDIO_OUTPUT_HF,
        AUDIO_OUTPUT_OPENAI_REALTIME,
        AUDIO_OUTPUT_CHATTERBOX,
        AUDIO_OUTPUT_ELEVENLABS,
    ],
)
def test_get_session_instructions_non_tts_outputs_strip(
    audio_output_backend: str, tmp_path: Path
) -> None:
    """All non-Gemini-TTS outputs strip the delivery-tags section."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(audio_output_backend=audio_output_backend)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER not in result
