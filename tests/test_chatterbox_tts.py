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
        "message": {
            "role": "assistant",
            "content": "Hey there!",
            "tool_calls": [],
        }
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
# _parse_text_tool_args
# ---------------------------------------------------------------------------


def test_parse_text_tool_args_json_dict() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args('{"action": "scan", "name": "Alice"}')
    assert result == {"action": "scan", "name": "Alice"}


def test_parse_text_tool_args_value_with_comma() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args('{"message": "hello, world"}')
    assert result == {"message": "hello, world"}


def test_parse_text_tool_args_bare_kv_fallback() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args("action: scan")
    assert result == {"action": "scan"}


def test_parse_text_tool_args_empty_string() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args("")
    assert result == {}


# ---------------------------------------------------------------------------
# _parse_json_content_tool_call
# ---------------------------------------------------------------------------


def test_parse_json_content_openai_style() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    text = '{"function": {"name": "greet", "arguments": {"action": "scan"}}}'
    result = _parse_json_content_tool_call(text)
    assert result == ("greet", {"action": "scan"})


def test_parse_json_content_flat_style() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    text = '{"name": "play_emotion", "arguments": {"emotion": "laughing1"}}'
    result = _parse_json_content_tool_call(text)
    assert result == ("play_emotion", {"emotion": "laughing1"})


def test_parse_json_content_flat_style_no_arguments_key() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    # {"name": "greet"} with no arguments key — not a tool call after guard fix
    text = '{"name": "greet"}'
    result = _parse_json_content_tool_call(text)
    assert result is None


def test_parse_json_content_returns_none_for_json_with_name_but_no_arguments() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    # A JSON object with "name" but no "arguments" key is not a tool call
    assert _parse_json_content_tool_call('{"name": "Rick", "age": 65}') is None


def test_parse_json_content_returns_none_for_plain_text() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    assert _parse_json_content_tool_call("Hello, how are you?") is None


def test_parse_json_content_returns_none_for_invalid_json() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    assert _parse_json_content_tool_call("{not valid json}") is None


def test_parse_json_content_returns_none_for_json_without_name() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    assert _parse_json_content_tool_call('{"foo": "bar"}') is None


def test_parse_json_content_string_serialized_arguments() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    # Hermes3 sometimes serializes arguments as a JSON string instead of a dict
    text = '{"function": {"name": "greet", "arguments": "{\\"action\\": \\"scan\\"}"}}'
    result = _parse_json_content_tool_call(text)
    assert result == ("greet", {"action": "scan"})


@pytest.mark.asyncio
async def test_call_llm_detects_json_content_tool_call() -> None:
    """_call_llm dispatches a JSON-format tool call found in the content field."""
    import httpx

    handler = _make_handler()

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "message": {
            "content": '{"function": {"name": "greet", "arguments": {"action": "scan"}}}',
            "tool_calls": [],
        }
    }
    handler._http.post = AsyncMock(return_value=fake_resp)

    text, tool_calls, _ = await handler._call_llm()

    assert text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "greet"
    assert tool_calls[0]["function"]["arguments"] == {"action": "scan"}


@pytest.mark.asyncio
async def test_call_llm_injects_tool_use_addendum() -> None:
    """The system message sent to Ollama includes the tool-use addendum."""
    import httpx
    from robot_comic.chatterbox_tts import _TOOL_USE_ADDENDUM

    handler = _make_handler()
    captured_payloads: list[dict] = []

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"message": {"content": "Hi!", "tool_calls": []}}

    async def capturing_post(url, *, json=None, **kwargs):
        captured_payloads.append(json or {})
        return fake_resp

    handler._http.post = capturing_post  # type: ignore[method-assign]

    await handler._call_llm()

    assert captured_payloads, "No LLM call was made"
    messages = captured_payloads[0]["messages"]
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    assert system_msg is not None, "No system message found in LLM payload"
    assert _TOOL_USE_ADDENDUM in system_msg["content"]


# ---------------------------------------------------------------------------
# Retry-with-nudge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nudge_fires_on_empty_response() -> None:
    """When LLM returns empty text and no tool calls, a nudge is sent and its tool call returned."""
    handler = _make_handler()
    nudge_detected = False

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal nudge_detected
        msgs = json.get("messages", [])
        if any(m.get("content") == "Please use a tool call now." for m in msgs):
            nudge_detected = True
            resp_data = {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "greet", "arguments": {"action": "scan"}}}],
                }
            }
        else:
            resp_data = {"message": {"content": "", "tool_calls": []}}
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = resp_data
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls, _ = await handler._call_llm()

    assert nudge_detected
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "greet"


@pytest.mark.asyncio
async def test_nudge_fires_at_most_once() -> None:
    """Even if nudge response is also empty, only one nudge is sent."""
    handler = _make_handler()
    nudge_count = 0

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal nudge_count
        msgs = json.get("messages", [])
        if any(m.get("content") == "Please use a tool call now." for m in msgs):
            nudge_count += 1
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "", "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls, _ = await handler._call_llm()

    assert nudge_count == 1
    assert text == ""
    assert tool_calls == []


@pytest.mark.asyncio
async def test_nudge_not_fired_when_text_present() -> None:
    """No nudge when the response contains meaningful text."""
    handler = _make_handler()
    call_count = 0

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "Hello there!", "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls, _ = await handler._call_llm()

    assert call_count == 1
    assert text == "Hello there!"


@pytest.mark.asyncio
async def test_nudge_not_fired_when_tool_calls_present() -> None:
    """No nudge when the response already contains tool calls."""
    handler = _make_handler()
    call_count = 0

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "greet", "arguments": {}}}],
            }
        }
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    _, tool_calls, _raw = await handler._call_llm()

    assert call_count == 1
    assert len(tool_calls) == 1


@pytest.mark.asyncio
async def test_nudge_does_not_modify_conversation_history() -> None:
    """The nudge message is ephemeral — not saved to _conversation_history."""
    handler = _make_handler()
    handler._conversation_history = [{"role": "user", "content": "hello"}]
    history_before = list(handler._conversation_history)

    async def fake_post(url, *, json=None, **kwargs):
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "", "tool_calls": []}}
        return fake_resp

    history_ref = handler._conversation_history
    handler._http.post = fake_post  # type: ignore[method-assign]
    await handler._call_llm()

    assert handler._conversation_history is history_ref
    assert handler._conversation_history == history_before


@pytest.mark.asyncio
async def test_nudge_recovers_text_format_tool_call() -> None:
    """Nudge response containing a text-format tool call is correctly parsed."""
    handler = _make_handler()

    async def fake_post(url, *, json=None, **kwargs):
        msgs = json.get("messages", [])
        if any(m.get("content") == "Please use a tool call now." for m in msgs):
            content = "{function:dance}"
        else:
            content = ""
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": content, "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls, _ = await handler._call_llm()

    assert text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "dance"


@pytest.mark.asyncio
async def test_nudge_recovers_json_content_tool_call() -> None:
    """Nudge response containing a JSON-format tool call in content is correctly parsed."""
    handler = _make_handler()

    async def fake_post(url, *, json=None, **kwargs):
        msgs = json.get("messages", [])
        if any(m.get("content") == "Please use a tool call now." for m in msgs):
            content = '{"name": "play_emotion", "arguments": {"emotion": "laughing1"}}'
        else:
            content = ""
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": content, "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls, _ = await handler._call_llm()

    assert text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "play_emotion"


# ---------------------------------------------------------------------------
# _trim_tool_spec — enum and description trimming
# ---------------------------------------------------------------------------

def _make_spec(
    *,
    name: str = "my_tool",
    description: str = "A" * 200,
    enum: list | None = None,
    prop_description: str = "B" * 200,
) -> dict:
    props: dict = {"param": {"type": "string", "description": prop_description}}
    if enum is not None:
        props["param"]["enum"] = enum
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": props},
    }


def test_trim_tool_spec_truncates_top_level_description() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    spec = _make_spec(description="X" * 300)
    result = ChatterboxTTSResponseHandler._trim_tool_spec(spec)
    assert len(result["function"]["description"]) == 80


def test_trim_tool_spec_truncates_property_description() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    spec = _make_spec(prop_description="Y" * 200)
    result = ChatterboxTTSResponseHandler._trim_tool_spec(spec)
    prop = result["function"]["parameters"]["properties"]["param"]
    assert len(prop["description"]) == 50


def test_trim_tool_spec_truncates_large_enum() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    big_enum = [f"emotion{i}" for i in range(81)]
    spec = _make_spec(enum=big_enum)
    result = ChatterboxTTSResponseHandler._trim_tool_spec(spec)
    prop = result["function"]["parameters"]["properties"]["param"]
    assert len(prop["enum"]) == ChatterboxTTSResponseHandler._MAX_ENUM_VALUES
    assert prop["enum"] == big_enum[: ChatterboxTTSResponseHandler._MAX_ENUM_VALUES]


def test_trim_tool_spec_preserves_small_enum() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    small_enum = ["a", "b", "c"]
    spec = _make_spec(enum=small_enum)
    result = ChatterboxTTSResponseHandler._trim_tool_spec(spec)
    prop = result["function"]["parameters"]["properties"]["param"]
    assert prop["enum"] == small_enum


def test_trim_tool_spec_no_enum_untouched() -> None:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    spec = _make_spec()
    result = ChatterboxTTSResponseHandler._trim_tool_spec(spec)
    prop = result["function"]["parameters"]["properties"]["param"]
    assert "enum" not in prop
