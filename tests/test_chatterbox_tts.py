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

    async def fake_llm() -> tuple[str, list, dict]:
        return "Hello! How are you today?", [], {}

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

    async def fake_llm() -> tuple[str, list, dict]:
        return "First sentence. Second sentence.", [], {}

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

    async def fake_llm() -> tuple[str, list, dict]:
        return "Hiya!", [], {}

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

    async def fake_llm() -> tuple[str, list, dict]:
        return "Hello. World.", [], {}

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

    async def fake_llm() -> tuple[str, list, dict]:
        return "Hello. World.", [], {}

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


@pytest.mark.asyncio
async def test_call_llm_returns_raw_message() -> None:
    """_call_llm returns a 3-tuple; the third element is the raw assistant message dict."""
    import httpx
    handler = _make_handler()

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hey there!", "tool_calls": []}}]
    }
    handler._http.post = AsyncMock(return_value=fake_resp)

    result = await handler._call_llm()

    assert len(result) == 3
    text, tool_calls, raw_message = result
    assert text == "Hey there!"
    assert raw_message["role"] == "assistant"
    assert "content" in raw_message


@pytest.mark.asyncio
async def test_start_tool_calls_returns_bg_tools() -> None:
    """_start_tool_calls returns (call_id, BackgroundTool) pairs, one per tool call."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()

    async def fake_start_tool(call_id, tool_call_routine, is_idle_tool_call):
        bg = BackgroundTool(
            id=call_id,
            tool_name=tool_call_routine.tool_name,
            is_idle_tool_call=False,
            status=ToolState.RUNNING,
        )
        return bg

    object.__setattr__(handler.tool_manager, "start_tool", fake_start_tool)  # type: ignore[misc]

    tool_calls = [
        {"function": {"name": "dance", "arguments": {"style": "wave"}}},
        {"function": {"name": "play_emotion", "arguments": {"emotion": "happy1"}}},
    ]
    result = await handler._start_tool_calls(tool_calls)

    assert len(result) == 2
    for call_id, bg_tool in result:
        assert isinstance(call_id, str) and len(call_id) == 8
        assert isinstance(bg_tool, BackgroundTool)
    assert {bg.tool_name for _, bg in result} == {"dance", "play_emotion"}


# ---------------------------------------------------------------------------
# Two-phase tool result feedback (end-to-end)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_llm_pass_fires_on_meaningful_result() -> None:
    """Camera returns a long description → two LLM calls, two TTS calls."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    tts_texts: list[str] = []
    camera_result = {
        "description": "A young woman with curly red hair stands close to the camera, grinning wide."
    }

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="cam1",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result=camera_result,
    )
    bg_tool._task = asyncio.create_task(instant_task())

    call_count = 0

    async def patched_llm():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                "Let me take a look!",
                [{"function": {"name": "camera", "arguments": {}}}],
                {"role": "assistant", "content": "Let me take a look!",
                 "tool_calls": [{"function": {"name": "camera", "arguments": {}}}]},
            )
        return "Oh yeah, I see a grinning woman!", [], {"role": "assistant", "content": "Oh yeah, I see a grinning woman!"}

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        tts_texts.append(text)
        return _pcm_bytes(2400)

    async def fake_start_tool_calls(tool_calls):
        return [("cam1", bg_tool)]

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("what do you see?")

    assert call_count == 2, f"Expected 2 LLM calls, got {call_count}"
    assert len(tts_texts) >= 2


@pytest.mark.asyncio
async def test_no_second_pass_for_action_tools() -> None:
    """Dance returns {} → only one LLM call is made."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    llm_call_count = 0

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="dance1",
        tool_name="dance",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={},
    )
    bg_tool._task = asyncio.create_task(instant_task())

    async def patched_llm():
        nonlocal llm_call_count
        llm_call_count += 1
        return (
            "I'll dance for you!",
            [{"function": {"name": "dance", "arguments": {}}}],
            {"role": "assistant", "content": "I'll dance for you!",
             "tool_calls": [{"function": {"name": "dance", "arguments": {}}}]},
        )

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        return _pcm_bytes(2400)

    async def fake_start_tool_calls(tool_calls):
        return [("dance1", bg_tool)]

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("dance!")

    assert llm_call_count == 1


@pytest.mark.asyncio
async def test_tool_message_appended_to_history() -> None:
    """A role=tool message with tool_call_id appears in history before the second LLM call."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    messages_seen_on_second_call: list[dict] = []
    call_count = 0
    camera_result = {
        "description": "An older gentleman in a Hawaiian shirt waves at the camera enthusiastically."
    }

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="cam2",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result=camera_result,
    )
    bg_tool._task = asyncio.create_task(instant_task())

    async def patched_llm():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                "Checking the camera!",
                [{"function": {"name": "camera", "arguments": {}}}],
                {"role": "assistant", "content": "Checking the camera!",
                 "tool_calls": [{"function": {"name": "camera", "arguments": {}}}]},
            )
        messages_seen_on_second_call.extend(list(handler._conversation_history))
        return "I see someone waving!", [], {"role": "assistant", "content": "I see someone waving!"}

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        return _pcm_bytes(2400)

    async def fake_start_tool_calls(tool_calls):
        return [("cam2", bg_tool)]

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("who's there?")

    tool_messages = [m for m in messages_seen_on_second_call if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "cam2"
    import json as _json
    content = _json.loads(tool_messages[0]["content"])
    assert "description" in content


@pytest.mark.asyncio
async def test_assistant_message_preserves_tool_calls_field() -> None:
    """When Ollama returns tool_calls, the history entry has a tool_calls field."""
    handler = _make_handler()

    async def patched_llm():
        raw_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "dance", "arguments": {}}}],
        }
        return "", [{"function": {"name": "dance", "arguments": {}}}], raw_msg

    async def fake_start_tool_calls(tool_calls):
        return []

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("dance!")

    assistant_entries = [m for m in handler._conversation_history if m.get("role") == "assistant"]
    assert len(assistant_entries) == 1
    assert "tool_calls" in assistant_entries[0]
    assert assistant_entries[0]["tool_calls"][0]["function"]["name"] == "dance"


def _drain_queue(q: asyncio.Queue) -> list:
    items = []
    while not q.empty():
        try:
            items.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# ---------------------------------------------------------------------------
# _wav_to_pcm gain parameter
# ---------------------------------------------------------------------------


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def test_wav_to_pcm_gain_doubles_amplitude() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    samples = np.full(100, 1000, dtype=np.int16)
    wav = _make_wav_bytes(samples)

    pcm = ChatterboxTTSResponseHandler._wav_to_pcm(wav, gain=2.0)
    out = np.frombuffer(pcm, dtype=np.int16)

    assert np.all(out == 2000)


def test_wav_to_pcm_gain_clips_at_int16_max() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    samples = np.full(100, 20000, dtype=np.int16)
    wav = _make_wav_bytes(samples)

    pcm = ChatterboxTTSResponseHandler._wav_to_pcm(wav, gain=2.0)
    out = np.frombuffer(pcm, dtype=np.int16)

    assert np.all(out == 32767)


def test_wav_to_pcm_default_gain_is_one() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    samples = np.full(100, 5000, dtype=np.int16)
    wav = _make_wav_bytes(samples)

    pcm = ChatterboxTTSResponseHandler._wav_to_pcm(wav)
    out = np.frombuffer(pcm, dtype=np.int16)

    assert np.all(out == 5000)


@pytest.mark.asyncio
async def test_handler_applies_chatterbox_gain_from_config() -> None:
    """Handler reads CHATTERBOX_GAIN from config and passes it to _wav_to_pcm."""
    import io, wave
    from unittest.mock import patch

    handler = _make_handler()

    samples = np.full(2400, 8000, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(samples.tobytes())
    wav_bytes = buf.getvalue()

    import httpx
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.raise_for_status = MagicMock()
    fake_response.content = wav_bytes

    async def fake_post(url, *, json=None, **kwargs):
        return fake_response

    handler._http.post = fake_post  # type: ignore[method-assign]

    with patch("robot_comic.chatterbox_tts.config") as mock_cfg:
        mock_cfg.CHATTERBOX_GAIN = 2.0
        mock_cfg.CHATTERBOX_VOICE = "don_rickles"
        mock_cfg.CHATTERBOX_EXAGGERATION = 1.0
        mock_cfg.CHATTERBOX_CFG_WEIGHT = 0.5
        mock_cfg.CHATTERBOX_TEMPERATURE = 0.6
        mock_cfg.CHATTERBOX_URL = "http://localhost:8004"
        pcm = await handler._call_chatterbox_tts("Hello")

    assert pcm is not None
    out = np.frombuffer(pcm, dtype=np.int16)
    assert np.all(out == 16000)


# ---------------------------------------------------------------------------
# _is_meaningful_result  (formerly preceded by removed Ollama-era sections:
# _parse_text_tool_args, _parse_json_content, nudge, _trim_tool_spec —
# all removed when Ollama backend was replaced with llama-server)
# ---------------------------------------------------------------------------


def test_meaningful_result_camera_passes() -> None:
    """A result with a long string value passes the meaningful filter."""
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    result = {
        "description": "A person is standing in the center of the frame, looking directly at the camera."
    }
    assert ChatterboxTTSResponseHandler._is_meaningful_result(result) is True


def test_meaningful_result_action_filtered() -> None:
    """Empty dict or short-value dict does not pass the meaningful filter."""
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    assert ChatterboxTTSResponseHandler._is_meaningful_result({}) is False
    assert ChatterboxTTSResponseHandler._is_meaningful_result({"status": "ok"}) is False
    assert ChatterboxTTSResponseHandler._is_meaningful_result({"status": "done", "count": 3}) is False


# ---------------------------------------------------------------------------
# _await_tool_results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_await_tool_results_returns_completed() -> None:
    """Completed tool tasks have their results returned."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    expected_result = {
        "description": "A smiling person stands before a plain background, facing the camera."
    }

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="abc123",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result=expected_result,
    )
    bg_tool._task = asyncio.create_task(instant_task())

    results = await handler._await_tool_results([("abc123", bg_tool)], timeout=1.0)

    assert results == {"abc123": expected_result}


@pytest.mark.asyncio
async def test_await_tool_results_timeout_excluded() -> None:
    """A tool that doesn't finish within the timeout is excluded from results."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()

    async def never_finishes() -> None:
        await asyncio.sleep(9999)

    bg_tool = BackgroundTool(
        id="slow1",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.RUNNING,
    )
    bg_tool._task = asyncio.create_task(never_finishes())

    results = await handler._await_tool_results([("slow1", bg_tool)], timeout=0.05)

    assert results == {}

    bg_tool._task.cancel()
    try:
        await bg_tool._task
    except (asyncio.CancelledError, Exception):
        pass
