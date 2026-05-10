"""Tests for ChatterboxTTSResponseHandler and sentence pipelining."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _make_handler():
    from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler

    deps = _make_deps()
    handler = LocalSTTChatterboxHandler(deps)
    handler._http = AsyncMock()
    return handler


def _pcm_bytes(n_samples: int = 2400) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_on_period() -> None:
    from robot_comic.chatterbox_tts import _split_sentences

    result = _split_sentences("Hello there. How are you?")
    assert result == ["Hello there.", "How are you?"]


def test_split_sentences_on_exclamation() -> None:
    from robot_comic.chatterbox_tts import _split_sentences

    result = _split_sentences("Hello! It is great to see you. How are you doing today?")
    assert result == ["Hello!", "It is great to see you.", "How are you doing today?"]


def test_split_sentences_single_sentence() -> None:
    from robot_comic.chatterbox_tts import _split_sentences

    result = _split_sentences("No split needed here")
    assert result == ["No split needed here"]


def test_split_sentences_empty_string() -> None:
    from robot_comic.chatterbox_tts import _split_sentences

    result = _split_sentences("")
    assert result == []


def test_split_sentences_strips_whitespace() -> None:
    from robot_comic.chatterbox_tts import _split_sentences

    result = _split_sentences("  Hello.   Goodbye.  ")
    assert result == ["Hello.", "Goodbye."]


# ---------------------------------------------------------------------------
# Sentence pipelining in _dispatch_completed_transcript
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_called_once_per_sentence() -> None:
    """With a two-sentence response, TTS is called twice (once per sentence)."""
    handler = _make_handler()

    tts_texts: list[str] = []

    async def fake_llm() -> tuple[str, list]:
        return "Hello! How are you today?", []

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        tts_texts.append(text)
        return _pcm_bytes(2400)

    handler._call_llm = fake_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("hi")

    assert len(tts_texts) == 2
    assert tts_texts[0] == "Hello!"
    assert tts_texts[1] == "How are you today?"


@pytest.mark.asyncio
async def test_frames_emitted_per_sentence_not_buffered() -> None:
    """Frames from sentence 1 appear in the queue before sentence 2 TTS even runs."""
    handler = _make_handler()

    sentence1_done = asyncio.Event()
    tts_call_order: list[str] = []
    queue_snapshots: dict[str, int] = {}

    async def fake_llm() -> tuple[str, list]:
        return "First sentence. Second sentence.", []

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        tts_call_order.append(text)
        if "First" in text:
            sentence1_done.set()
        else:
            # By the time second TTS is called, queue should already have frames
            queue_snapshots["before_second_tts"] = handler.output_queue.qsize()
        return _pcm_bytes(2400)

    handler._call_llm = fake_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("go")

    # Sentence 1 frames were queued before sentence 2 TTS was called
    assert queue_snapshots.get("before_second_tts", 0) > 0


@pytest.mark.asyncio
async def test_single_sentence_still_produces_audio() -> None:
    """A response with no sentence boundary still generates audio frames."""
    handler = _make_handler()

    async def fake_llm() -> tuple[str, list]:
        return "Hiya!", []

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        return _pcm_bytes(4800)

    handler._call_llm = fake_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("hey")

    audio_frames = [
        item for item in _drain_queue(handler.output_queue) if isinstance(item, tuple)
    ]
    assert len(audio_frames) == 2  # 4800 samples / 2400 per frame


@pytest.mark.asyncio
async def test_tts_error_on_all_sentences_pushes_error_output() -> None:
    """When every TTS call returns None, an error AdditionalOutputs is queued."""
    handler = _make_handler()

    async def fake_llm() -> tuple[str, list]:
        return "Hello. World.", []

    async def failing_tts(text: str, *, exaggeration=None, cfg_weight=None) -> None:
        return None

    handler._call_llm = fake_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = failing_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("go")

    items = _drain_queue(handler.output_queue)
    error_items = [
        i for i in items
        if isinstance(i, AdditionalOutputs) and "error" in str(i.args).lower()
    ]
    assert len(error_items) >= 1


@pytest.mark.asyncio
async def test_split_text_disabled_in_tts_payload() -> None:
    """TTS requests have split_text=False since we split sentences client-side."""
    handler = _make_handler()
    captured_payloads: list[dict] = []

    async def fake_llm() -> tuple[str, list]:
        return "Hello. World.", []

    # Intercept the actual HTTP call to inspect the payload
    import httpx

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.raise_for_status = MagicMock()

    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(_pcm_bytes(2400))
    wav_bytes = buf.getvalue()

    fake_response.content = wav_bytes
    handler._http.post = AsyncMock(return_value=fake_response)  # type: ignore[method-assign]

    async def capturing_post(url, *, json=None, **kwargs):
        captured_payloads.append(json or {})
        return fake_response

    handler._http.post = capturing_post  # type: ignore[method-assign]

    handler._call_llm = fake_llm  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("go")

    assert len(captured_payloads) >= 1
    for payload in captured_payloads:
        assert payload.get("split_text") is False


def _drain_queue(q: asyncio.Queue) -> list:
    items = []
    while not q.empty():
        try:
            items.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items
