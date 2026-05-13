"""Unit tests for backend-aware GEMINI TTS DELIVERY TAGS section filtering.

Covers Issue #102 — the ``_filter_delivery_tags`` helper in ``prompts.py``
should include or strip the delivery-tags section depending on the active
backend, and an env-var override must bypass the stripping.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from robot_comic.config import (
    HF_BACKEND,
    GEMINI_BACKEND,
    OPENAI_BACKEND,
    CHATTERBOX_OUTPUT,
    GEMINI_TTS_OUTPUT,
    LOCAL_STT_BACKEND,
    LLAMA_GEMINI_TTS_OUTPUT,
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
    backend: str,
    local_stt_response: str = OPENAI_BACKEND,
    force_delivery_tags: bool = False,
) -> object:
    """Return a minimal config-like namespace for patching."""

    class _FakeConfig:
        BACKEND_PROVIDER = backend
        LOCAL_STT_RESPONSE_BACKEND = local_stt_response
        FORCE_DELIVERY_TAGS = force_delivery_tags

    return _FakeConfig()


# ---------------------------------------------------------------------------
# _uses_gemini_tts
# ---------------------------------------------------------------------------


def test_uses_gemini_tts_output_is_true() -> None:
    """gemini_tts backend must return True."""
    assert _uses_gemini_tts(GEMINI_TTS_OUTPUT, OPENAI_BACKEND) is True


def test_uses_llama_gemini_tts_output_is_true() -> None:
    """llama_gemini_tts backend must return True."""
    assert _uses_gemini_tts(LLAMA_GEMINI_TTS_OUTPUT, OPENAI_BACKEND) is True


def test_uses_gemini_live_backend_is_false() -> None:
    """GEMINI_BACKEND ('gemini') = Gemini Live — does NOT consume TTS tags."""
    assert _uses_gemini_tts(GEMINI_BACKEND, OPENAI_BACKEND) is False


def test_uses_hf_backend_is_false() -> None:
    """HuggingFace backend does not consume TTS tags."""
    assert _uses_gemini_tts(HF_BACKEND, OPENAI_BACKEND) is False


def test_uses_openai_backend_is_false() -> None:
    """OpenAI backend does not consume TTS tags."""
    assert _uses_gemini_tts(OPENAI_BACKEND, OPENAI_BACKEND) is False


def test_uses_chatterbox_backend_is_false() -> None:
    """Chatterbox backend does not consume TTS tags."""
    assert _uses_gemini_tts(CHATTERBOX_OUTPUT, OPENAI_BACKEND) is False


def test_uses_local_stt_with_gemini_tts_response_is_true() -> None:
    """Local STT + gemini_tts response backend must return True."""
    assert _uses_gemini_tts(LOCAL_STT_BACKEND, GEMINI_TTS_OUTPUT) is True


def test_uses_local_stt_with_llama_gemini_tts_response_is_true() -> None:
    """Local STT + llama_gemini_tts response backend must return True."""
    assert _uses_gemini_tts(LOCAL_STT_BACKEND, LLAMA_GEMINI_TTS_OUTPUT) is True


def test_uses_local_stt_with_openai_response_is_false() -> None:
    """Local STT + openai response backend must return False."""
    assert _uses_gemini_tts(LOCAL_STT_BACKEND, OPENAI_BACKEND) is False


def test_uses_local_stt_with_chatterbox_response_is_false() -> None:
    """Local STT + chatterbox response backend must return False."""
    assert _uses_gemini_tts(LOCAL_STT_BACKEND, CHATTERBOX_OUTPUT) is False


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


def test_filter_gemini_tts_backend_keeps_section() -> None:
    """gemini_tts backend keeps the delivery-tags section."""
    fake = _make_fake_config(backend=GEMINI_TTS_OUTPUT)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_llama_gemini_tts_backend_keeps_section() -> None:
    """llama_gemini_tts backend keeps the delivery-tags section."""
    fake = _make_fake_config(backend=LLAMA_GEMINI_TTS_OUTPUT)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_gemini_live_backend_strips_section() -> None:
    """Gemini Live backend strips the delivery-tags section."""
    fake = _make_fake_config(backend=GEMINI_BACKEND)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_hf_backend_strips_section() -> None:
    """HuggingFace backend strips the delivery-tags section."""
    fake = _make_fake_config(backend=HF_BACKEND)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_openai_backend_strips_section() -> None:
    """OpenAI backend strips the delivery-tags section."""
    fake = _make_fake_config(backend=OPENAI_BACKEND)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_chatterbox_backend_strips_section() -> None:
    """Chatterbox backend strips the delivery-tags section."""
    fake = _make_fake_config(backend=CHATTERBOX_OUTPUT)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_local_stt_gemini_tts_response_keeps_section() -> None:
    """Local STT + gemini_tts response keeps the delivery-tags section."""
    fake = _make_fake_config(backend=LOCAL_STT_BACKEND, local_stt_response=GEMINI_TTS_OUTPUT)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_local_stt_chatterbox_response_strips_section() -> None:
    """Local STT + chatterbox response strips the delivery-tags section."""
    fake = _make_fake_config(backend=LOCAL_STT_BACKEND, local_stt_response=CHATTERBOX_OUTPUT)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_force_override_keeps_section_on_chatterbox() -> None:
    """REACHY_MINI_FORCE_DELIVERY_TAGS=1 bypasses stripping on Chatterbox."""
    fake = _make_fake_config(backend=CHATTERBOX_OUTPUT, force_delivery_tags=True)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_force_override_keeps_section_on_hf() -> None:
    """REACHY_MINI_FORCE_DELIVERY_TAGS=1 bypasses stripping on HuggingFace."""
    fake = _make_fake_config(backend=HF_BACKEND, force_delivery_tags=True)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER in result


def test_filter_force_false_does_not_keep_section_on_chatterbox() -> None:
    """FORCE_DELIVERY_TAGS=False still strips on Chatterbox."""
    fake = _make_fake_config(backend=CHATTERBOX_OUTPUT, force_delivery_tags=False)
    with patch("robot_comic.prompts.config", fake):
        result = _filter_delivery_tags(_SAMPLE_INSTRUCTIONS)
    assert _SECTION_HEADER not in result


def test_filter_stripped_result_preserves_live_guidance() -> None:
    """Stripping TTS section leaves GEMINI LIVE DELIVERY GUIDANCE intact."""
    fake = _make_fake_config(backend=GEMINI_BACKEND)
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
    fake_cfg = _make_fake_config(backend=CHATTERBOX_OUTPUT)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER not in result
    assert "## IDENTITY" in result


def test_get_session_instructions_gemini_tts_keeps_section(tmp_path: Path) -> None:
    """get_session_instructions() keeps the delivery section for Gemini TTS."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(backend=GEMINI_TTS_OUTPUT)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER in result


def test_get_session_instructions_force_override_keeps_section_on_hf(tmp_path: Path) -> None:
    """get_session_instructions() respects FORCE_DELIVERY_TAGS on HuggingFace."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(backend=HF_BACKEND, force_delivery_tags=True)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER in result


@pytest.mark.parametrize(
    "backend",
    [GEMINI_BACKEND, HF_BACKEND, OPENAI_BACKEND, CHATTERBOX_OUTPUT],
)
def test_get_session_instructions_non_tts_backends_strip(backend: str, tmp_path: Path) -> None:
    """All non-Gemini-TTS backends strip the delivery-tags section."""
    profiles_root = _write_test_profile(tmp_path)
    fake_cfg = _make_fake_config(backend=backend)
    fake_cfg.REACHY_MINI_CUSTOM_PROFILE = "test_persona"  # type: ignore[attr-defined]
    fake_cfg.PROFILES_DIRECTORY = profiles_root  # type: ignore[attr-defined]
    with patch("robot_comic.prompts.config", fake_cfg):
        result = get_session_instructions()
    assert _SECTION_HEADER not in result
