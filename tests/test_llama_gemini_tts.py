"""Tests for LlamaGeminiTTSResponseHandler — focus on the TTS override.

The LLM/tool dispatch path is inherited from BaseLlamaResponseHandler and is
already covered by tests/test_llama_base.py. These tests target the parts
unique to the Gemini-TTS variant: _synthesize_and_enqueue, _call_gemini_tts,
voice management, and copy() preserving deps.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _make_handler():
    from robot_comic.llama_gemini_tts import LocalSTTLlamaGeminiTTSHandler

    deps = _make_deps()
    handler = LocalSTTLlamaGeminiTTSHandler(deps)
    handler._http = AsyncMock()
    handler._client = MagicMock()
    return handler


def _encoded_pcm(n_samples: int = 2400) -> str:
    raw = np.zeros(n_samples, dtype=np.int16).tobytes()
    return base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# _synthesize_and_enqueue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_calls_gemini_tts_once_per_sentence() -> None:
    """A two-sentence response triggers two TTS calls (sentence-level pipelining)."""
    handler = _make_handler()
    captured: list[str] = []

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        captured.append(text)
        return np.zeros(2400, dtype=np.int16).tobytes()

    handler._call_gemini_tts = fake_tts  # type: ignore[method-assign]

    await handler._synthesize_and_enqueue("First sentence. Second sentence.")

    assert captured == ["First sentence.", "Second sentence."]


@pytest.mark.asyncio
async def test_synthesize_strips_tags_from_spoken_text() -> None:
    """Delivery tags are removed from the text actually sent to TTS."""
    handler = _make_handler()
    captured: list[str] = []

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        captured.append(text)
        return np.zeros(2400, dtype=np.int16).tobytes()

    handler._call_gemini_tts = fake_tts  # type: ignore[method-assign]

    await handler._synthesize_and_enqueue("[fast] You hockey puck!")

    assert captured == ["You hockey puck!"]


@pytest.mark.asyncio
async def test_synthesize_passes_tag_cues_in_system_instruction() -> None:
    """Tags in a sentence are translated into a 'Delivery cues' suffix for that call."""
    handler = _make_handler()
    captured_instructions: list[str | None] = []

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        captured_instructions.append(system_instruction)
        return np.zeros(2400, dtype=np.int16).tobytes()

    handler._call_gemini_tts = fake_tts  # type: ignore[method-assign]

    await handler._synthesize_and_enqueue("[annoyance] Look at this guy.")

    assert len(captured_instructions) == 1
    instr = captured_instructions[0]
    assert instr is not None
    assert "Delivery cues for this line:" in instr
    assert "exasperated" in instr


@pytest.mark.asyncio
async def test_synthesize_short_pause_inserts_silence_not_cue() -> None:
    """[short pause] yields real silence frames, not a TTS cue word."""
    handler = _make_handler()
    captured_instructions: list[str | None] = []

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        captured_instructions.append(system_instruction)
        return np.zeros(2400, dtype=np.int16).tobytes()

    handler._call_gemini_tts = fake_tts  # type: ignore[method-assign]

    await handler._synthesize_and_enqueue("First. [short pause] Second.")

    audio_frames = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, tuple):
            audio_frames.append(item)

    # 2 sentences × ≥1 frame each + silence burst (≥3 frames at 24kHz × 400ms / 2400 samples/frame).
    assert len(audio_frames) > 2
    # The "short pause" cue word must not leak into the TTS instruction.
    for instr in captured_instructions:
        assert instr is None or "short pause" not in instr


@pytest.mark.asyncio
async def test_synthesize_empty_response_returns_early() -> None:
    """An empty response_text emits nothing — no TTS call, no error item."""
    handler = _make_handler()
    handler._call_gemini_tts = AsyncMock()  # type: ignore[method-assign]

    await handler._synthesize_and_enqueue("")

    handler._call_gemini_tts.assert_not_called()
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_synthesize_all_tts_failures_pushes_error() -> None:
    """When every per-sentence TTS call returns None, an error AdditionalOutputs is queued."""
    handler = _make_handler()

    async def failing_tts(text: str, system_instruction: str | None = None) -> None:
        return None

    handler._call_gemini_tts = failing_tts  # type: ignore[method-assign]

    await handler._synthesize_and_enqueue("One. Two.")

    items: list[object] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    error_items = [i for i in items if isinstance(i, AdditionalOutputs) and "error" in str(i.args).lower()]
    assert error_items, f"Expected an error AdditionalOutputs, got {items!r}"


# ---------------------------------------------------------------------------
# _call_gemini_tts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_gemini_tts_uses_default_system_instruction_when_unset() -> None:
    """With no system_instruction kwarg, the call falls back to the profile/default base."""
    handler = _make_handler()
    captured_configs: list[object] = []

    async def fake_generate(model, contents, config):
        captured_configs.append(config)
        part = MagicMock()
        part.inline_data.data = _encoded_pcm()
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        return response

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = fake_generate

    pcm = await handler._call_gemini_tts("Hello there")

    assert pcm is not None
    assert len(captured_configs) == 1
    # Default model (`gemini-3.1-flash-tts-preview`) drops system_instruction;
    # the cue is prepended to contents instead. Accept either path.
    instr = captured_configs[0].system_instruction
    if instr is None:
        from robot_comic.gemini_tts import GEMINI_TTS_MODEL, _TTS_NO_SYSTEM_INSTRUCTION_MODELS

        assert GEMINI_TTS_MODEL in _TTS_NO_SYSTEM_INSTRUCTION_MODELS
    else:
        assert isinstance(instr, str) and instr  # non-empty


@pytest.mark.asyncio
async def test_call_gemini_tts_honors_explicit_system_instruction() -> None:
    """When the caller passes system_instruction, it is what reaches the SDK call."""
    handler = _make_handler()
    captured_configs: list[object] = []
    captured_contents: list[object] = []

    async def fake_generate(model, contents, config):
        captured_configs.append(config)
        captured_contents.append(contents)
        part = MagicMock()
        part.inline_data.data = _encoded_pcm()
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        return response

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = fake_generate

    await handler._call_gemini_tts("Hello", system_instruction="Whispered, conspiratorial.")

    # Default model drops system_instruction; the explicit cue is prepended to contents.
    instr = captured_configs[0].system_instruction
    if instr is None:
        assert any("Whispered, conspiratorial." in str(c) for c in captured_contents)
    else:
        assert instr == "Whispered, conspiratorial."


@pytest.mark.asyncio
async def test_call_gemini_tts_returns_none_after_all_failures() -> None:
    """All retries exhausted → returns None instead of raising."""
    handler = _make_handler()

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("500"))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await handler._call_gemini_tts("Hello")

    assert result is None


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_available_voices_returns_gemini_tts_voices() -> None:
    """The voice list is the Gemini TTS catalog, not the Live catalog or Chatterbox."""
    from robot_comic.config import GEMINI_TTS_AVAILABLE_VOICES

    handler = _make_handler()
    voices = await handler.get_available_voices()
    assert voices == list(GEMINI_TTS_AVAILABLE_VOICES)


def test_get_current_voice_falls_back_when_invalid() -> None:
    """A bogus override value falls back to the default voice."""
    from robot_comic.config import GEMINI_TTS_DEFAULT_VOICE

    handler = _make_handler()
    handler._voice_override = "NotARealVoice"
    assert handler.get_current_voice() == GEMINI_TTS_DEFAULT_VOICE


@pytest.mark.asyncio
async def test_change_voice_updates_override() -> None:
    """change_voice sets the override and is reflected in get_current_voice."""
    handler = _make_handler()
    await handler.change_voice("Puck")
    assert handler.get_current_voice() == "Puck"


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


def test_copy_preserves_deps_and_voice_override() -> None:
    """copy() returns a fresh handler with the same deps and voice override.

    Use a TTS-exclusive voice — voices shared with Gemini Live are deliberately
    not restored on copy() to keep the two backends' persisted voices separated.
    """
    handler = _make_handler()
    handler._voice_override = "Algenib"  # TTS-exclusive

    clone = handler.copy()

    assert clone.deps is handler.deps
    assert clone._voice_override == "Algenib"
    assert clone is not handler
