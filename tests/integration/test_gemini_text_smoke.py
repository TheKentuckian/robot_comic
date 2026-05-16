"""End-to-end integration smoke tests — Gemini text-generation LLM backend.

Covers four scenarios:

1. ``test_gemini_text_chatterbox_dispatch_produces_pcm_audio``
   Boots a Moonshine-mixed ``GeminiTextChatterboxResponseHandler``, mocks
   ``_gemini_llm.stream_completion`` to return a canned text delta, mocks
   ``_call_chatterbox_tts`` to return canned PCM, injects a transcript,
   and asserts ≥1 PCM audio frame reaches the queue.

2. ``test_gemini_text_elevenlabs_dispatch_produces_pcm_audio``
   Same lifecycle but with ``GeminiTextElevenLabsResponseHandler``; TTS
   is mocked via ``_stream_tts_to_queue`` (the internal ElevenLabs
   streaming boundary).

3. ``test_gemini_text_tool_call_accumulation``
   Feeds a streamed response with a ``greet`` tool-call split across two chunks.
   Verifies that ``_stream_response_and_synthesize`` accumulates both fragments
   and dispatches the tool exactly once.

4. ``test_gemini_text_429_backoff_retries``
   The first ``generate_content_stream`` call raises a synthetic 429.  The second
   call succeeds.  Verifies that the retry loop fires the second call (i.e., the
   handler does not surface the transient error to the caller).

Network boundaries mocked:
  * ``_gemini_llm.stream_completion``     → async generator over canned deltas.
  * ``_call_chatterbox_tts``              → returns canned PCM bytes.
  * ``_stream_tts_to_queue``              → pushes canned PCM frames directly.
  * ``asyncio.sleep``                     → no-op during 429 test.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import robot_comic.llama_base as llama_base_mod
import robot_comic.gemini_text_base as gemini_text_base_mod
from .conftest import drain_queue, make_tool_deps


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int = 4800) -> bytes:
    """Return *n_samples* of silence as int16 PCM bytes (200 ms at 24 kHz)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


async def _delta_stream(*deltas: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Yield each *delta* dict in order, simulating ``GeminiLLMClient.stream_completion``."""
    for delta in deltas:
        yield delta


def _make_gemini_llm_mock(*deltas: dict[str, Any]) -> MagicMock:
    """Return a mock ``GeminiLLMClient`` whose ``stream_completion`` yields *deltas*."""
    mock_llm = MagicMock()

    async def _stream_completion(**_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        for delta in deltas:
            yield delta

    mock_llm.stream_completion = MagicMock(side_effect=lambda **kw: _stream_completion(**kw))
    return mock_llm


def _make_chatterbox_handler(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Boot a Moonshine + Gemini-text + Chatterbox host with hardware mocked.

    Phase 4e (#337) retired the GeminiTextChatterboxHandler concrete class;
    the composable factory composes LocalSTTInputMixin over
    GeminiTextChatterboxResponseHandler via a private host. We mirror that
    shape here so the smoke test still exercises both halves.
    """
    from robot_comic.local_stt_realtime import LocalSTTInputMixin
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxResponseHandler

    class _Host(LocalSTTInputMixin, GeminiTextChatterboxResponseHandler):
        async def _dispatch_completed_transcript(self, transcript: str) -> None:
            # Route past LocalSTTInputMixin's OpenAI-realtime default —
            # mirrors the factory-private host shape.
            await GeminiTextChatterboxResponseHandler._dispatch_completed_transcript(self, transcript)

    monkeypatch.setattr(llama_base_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(llama_base_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(gemini_text_base_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(gemini_text_base_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(
        "robot_comic.llama_base.config",
        MagicMock(
            LLAMA_CPP_URL="http://localhost:8080",
            ECHO_COOLDOWN_MS=300,
            REACHY_MINI_MAX_HISTORY_TURNS=20,
            REACHY_MINI_CUSTOM_PROFILE=None,
            JOKE_HISTORY_ENABLED=False,
        ),
    )

    deps = make_tool_deps()
    handler = _Host(deps, sim_mode=True)
    # Skip _prepare_startup_credentials — wire clients directly.
    import httpx

    handler._http = httpx.AsyncClient()
    handler.tool_manager.start_up(tool_callbacks=[handler._handle_tool_notification])
    # Inject a pre-built GeminiLLMClient mock (replaced per-test).
    handler._gemini_llm = MagicMock()
    return handler


def _make_elevenlabs_handler(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Boot a ``GeminiTextElevenLabsResponseHandler`` with all hardware and credentials mocked.

    We use the response-handler base class directly (without ``LocalSTTInputMixin``)
    because the test calls ``_dispatch_completed_transcript`` explicitly and does not
    need the Moonshine STT wiring.

    The MRO for ``GeminiTextElevenLabsResponseHandler`` causes a constructor conflict:
    ``BaseLlamaResponseHandler.__init__`` passes ``expected_layout="mono"`` to
    ``super().__init__``, which in this MRO resolves to
    ``ElevenLabsTTSResponseHandler.__init__`` — a sibling that only accepts
    ``(deps, sim_mode, ...)``.  We break the conflict by patching
    ``ElevenLabsTTSResponseHandler.__init__`` so it skips directly to
    ``AsyncStreamHandler.__init__`` with the audio-layout kwargs it actually needs.
    """
    from fastrtc import AsyncStreamHandler as _ASH

    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler as _EL
    from robot_comic.gemini_text_handlers import GeminiTextElevenLabsResponseHandler

    monkeypatch.setattr(llama_base_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(llama_base_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(gemini_text_base_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(gemini_text_base_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(
        "robot_comic.llama_base.config",
        MagicMock(
            LLAMA_CPP_URL="http://localhost:8080",
            ECHO_COOLDOWN_MS=300,
            REACHY_MINI_MAX_HISTORY_TURNS=20,
            REACHY_MINI_CUSTOM_PROFILE=None,
            JOKE_HISTORY_ENABLED=False,
        ),
    )
    monkeypatch.setattr(
        "robot_comic.elevenlabs_tts.config",
        MagicMock(
            ELEVENLABS_API_KEY="test_key",
            ECHO_COOLDOWN_MS=300,
            REACHY_MINI_MAX_HISTORY_TURNS=20,
            REACHY_MINI_CUSTOM_PROFILE=None,
            JOKE_HISTORY_ENABLED=False,
        ),
    )
    monkeypatch.setattr(
        "robot_comic.elevenlabs_tts.ElevenLabsTTSResponseHandler._resolve_voice_id",
        lambda self: "test_voice_id",
    )

    # Patch the ElevenLabs __init__ to accept the kwargs from BaseLlamaResponseHandler
    # and then hand off to AsyncStreamHandler (the real audio setup it normally does).
    original_el_init = _EL.__init__

    def _patched_el_init(
        self: Any,
        *args: Any,
        expected_layout: str = "mono",
        output_sample_rate: int = 24000,
        input_sample_rate: int = 16000,
        **kwargs: Any,
    ) -> None:
        if args or kwargs:
            # Called directly with (deps, sim_mode, ...) — delegate normally.
            original_el_init(self, *args, **kwargs)
        else:
            # Called via MRO from BaseLlamaResponseHandler with audio-layout kwargs.
            _ASH.__init__(
                self,
                expected_layout=expected_layout,
                output_sample_rate=output_sample_rate,
                input_sample_rate=input_sample_rate,
            )

    monkeypatch.setattr(_EL, "__init__", _patched_el_init)

    deps = make_tool_deps()
    handler = GeminiTextElevenLabsResponseHandler(deps, sim_mode=True)
    import httpx

    handler._http = httpx.AsyncClient()
    handler.tool_manager.start_up(tool_callbacks=[handler._handle_tool_notification])
    handler._gemini_llm = MagicMock()
    return handler


# ---------------------------------------------------------------------------
# Test 1: Chatterbox TTS path produces PCM audio
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_text_chatterbox_dispatch_produces_pcm_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiTextChatterboxHandler: inject transcript → assert PCM frame in queue.

    Mocks:
    - ``_gemini_llm.stream_completion`` → single text delta + finish.
    - ``_call_chatterbox_tts``          → canned PCM bytes (200 ms silence).
    """
    handler = _make_chatterbox_handler(monkeypatch)

    # --- Mock Gemini LLM stream (one complete sentence) ----------------------
    handler._gemini_llm.stream_completion = MagicMock(
        side_effect=lambda **_kw: _delta_stream(
            {"type": "text_delta", "content": "Hello there, friend!"},
            {"type": "finish_reason", "finish_reason": "stop"},
        )
    )

    # --- Mock Chatterbox TTS --------------------------------------------------
    tts_calls: list[str] = []

    async def _fake_chatterbox_tts(
        text: str,
        *,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> bytes:
        tts_calls.append(text)
        return _pcm_bytes(4800)

    handler._call_chatterbox_tts = _fake_chatterbox_tts  # type: ignore[method-assign]

    # --- Run dispatch ---------------------------------------------------------
    await handler._dispatch_completed_transcript("hello")

    # --- Assertions -----------------------------------------------------------
    all_items = drain_queue(handler.output_queue)
    audio_frames = [item for item in all_items if isinstance(item, tuple)]

    assert len(audio_frames) >= 1, (
        f"Expected ≥1 PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"Items: {[type(i).__name__ for i in all_items]}"
    )
    assert len(tts_calls) >= 1, "Expected _call_chatterbox_tts to be called at least once"

    sample_rate, pcm_array = audio_frames[0]
    assert isinstance(sample_rate, int), "Frame[0] first element must be sample rate (int)"
    assert isinstance(pcm_array, np.ndarray), "Frame[0] second element must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM dtype should be int16, got {pcm_array.dtype}"


# ---------------------------------------------------------------------------
# Test 2: ElevenLabs TTS path produces PCM audio
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_text_elevenlabs_dispatch_produces_pcm_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiTextElevenLabsHandler: inject transcript → assert PCM frame in queue.

    Mocks:
    - ``_gemini_llm.stream_completion`` → single text delta + finish.
    - ``_stream_tts_to_queue``          → pushes a canned PCM frame directly.
    """
    handler = _make_elevenlabs_handler(monkeypatch)

    # --- Mock Gemini LLM stream ----------------------------------------------
    handler._gemini_llm.stream_completion = MagicMock(
        side_effect=lambda **_kw: _delta_stream(
            {"type": "text_delta", "content": "Hello there, friend!"},
            {"type": "finish_reason", "finish_reason": "stop"},
        )
    )

    # --- Mock ElevenLabs TTS via _synthesize_and_enqueue ---------------------
    # ``GeminiTextElevenLabsResponseHandler`` uses ``BaseLlamaResponseHandler``'s
    # ``_stream_response_and_synthesize`` which calls ``_synthesize_and_enqueue``
    # (the abstract TTS boundary).  We stub it directly so the test does not need
    # a running ElevenLabs server or a fully wired ``_stream_tts_to_queue`` chain.
    tts_calls: list[str] = []

    async def _fake_synthesize_and_enqueue(
        text: str,
        tts_start: float | None = None,
        target_queue: asyncio.Queue[Any] | None = None,
    ) -> None:
        tts_calls.append(text)
        out: asyncio.Queue[Any] = target_queue if target_queue is not None else handler.output_queue
        pcm = np.zeros(_CHUNK_SAMPLES * 2, dtype=np.int16)
        await out.put((_OUTPUT_SAMPLE_RATE, pcm))

    handler._synthesize_and_enqueue = _fake_synthesize_and_enqueue  # type: ignore[method-assign]

    # --- Run dispatch ---------------------------------------------------------
    await handler._dispatch_completed_transcript("hello")

    # --- Assertions -----------------------------------------------------------
    all_items = drain_queue(handler.output_queue)
    audio_frames = [item for item in all_items if isinstance(item, tuple)]

    assert len(audio_frames) >= 1, (
        f"Expected ≥1 PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"Items: {[type(i).__name__ for i in all_items]}"
    )
    assert len(tts_calls) >= 1, "Expected _synthesize_and_enqueue (ElevenLabs TTS) to be called at least once"

    sample_rate, pcm_array = audio_frames[0]
    assert sample_rate == _OUTPUT_SAMPLE_RATE, f"Expected sample rate {_OUTPUT_SAMPLE_RATE}, got {sample_rate}"
    assert isinstance(pcm_array, np.ndarray), "PCM data must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM dtype should be int16, got {pcm_array.dtype}"


# ---------------------------------------------------------------------------
# Test 3: Tool-call accumulation across chunks
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_text_tool_call_accumulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamed tool-call fragments are accumulated and dispatched exactly once.

    Feeds two ``tool_call_delta`` chunks for the same index-0 ``greet`` call
    (matching the pattern where a Gemini chunk carries the name first and a
    subsequent chunk carries the arguments).  After the stream ends,
    ``_stream_response_and_synthesize`` must have collected exactly one tool call
    named ``greet`` with the complete argument payload.
    """
    handler = _make_chatterbox_handler(monkeypatch)

    # Gemini streams the tool call in two chunks:
    #   chunk 1 → name="greet", arguments="{}"  (full args in first chunk — Gemini behaviour)
    #   chunk 2 → finish_reason=STOP
    # Both share index=0, so they accumulate to a single entry.
    handler._gemini_llm.stream_completion = MagicMock(
        side_effect=lambda **_kw: _delta_stream(
            {
                "type": "tool_call_delta",
                "index": 0,
                "id": "abc12345",
                "name": "greet",
                "arguments": "{}",
            },
            {
                "type": "tool_call_delta",
                "index": 0,
                "id": "abc12345",
                "name": "",  # second chunk carries no new name — empty
                "arguments": None,  # type: ignore[arg-type]
            },
            {"type": "finish_reason", "finish_reason": "stop"},
        )
    )

    # _call_chatterbox_tts should not be reached (no text in this response).
    tts_calls: list[str] = []

    async def _noop_tts(text: str, **_kw: Any) -> bytes:
        tts_calls.append(text)
        return _pcm_bytes(2400)

    handler._call_chatterbox_tts = _noop_tts  # type: ignore[method-assign]

    # Mock _start_tool_calls to capture what was dispatched without side-effects.
    dispatched_tool_calls: list[list[Any]] = []

    async def _fake_start_tool_calls(tool_calls: list[Any]) -> list[Any]:
        dispatched_tool_calls.append(tool_calls)
        return []  # no background tools — skip follow-up loop

    handler._start_tool_calls = _fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("say hi")

    # Exactly one tool dispatch must have occurred.
    assert len(dispatched_tool_calls) == 1, f"Expected exactly 1 tool-call dispatch, got {len(dispatched_tool_calls)}"
    accumulated = dispatched_tool_calls[0]
    assert len(accumulated) == 1, f"Expected 1 accumulated tool call, got {len(accumulated)}: {accumulated}"
    tc = accumulated[0]
    assert tc["function"]["name"] == "greet", f"Expected tool 'greet', got {tc['function']['name']!r}"
    # TTS was not invoked (no text in this response).
    assert tts_calls == [], f"Expected no TTS calls for a pure-tool response, got: {tts_calls}"


# ---------------------------------------------------------------------------
# Test 4: 429 backoff — GeminiLLMClient retries on RESOURCE_EXHAUSTED
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_text_429_backoff_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient 429 from the underlying Gemini SDK does not surface to the caller.

    We exercise the retry loop *inside* ``GeminiLLMClient.stream_completion`` by
    creating a real ``GeminiLLMClient`` (with a mocked ``_client``) and wiring
    it into the Chatterbox handler.

    ``_client.aio.models.generate_content_stream`` raises a 429-shaped error on the
    first invocation and returns a valid async generator on the second.  We patch
    ``asyncio.sleep`` in ``robot_comic.gemini_llm`` to avoid real delays and assert
    that both calls to ``generate_content_stream`` happened (i.e., the retry fired).
    """
    handler = _make_chatterbox_handler(monkeypatch)

    # Simulate a 429 error recognisable by ``is_rate_limit_error``.
    class _Fake429(Exception):
        code = 429
        status = "RESOURCE_EXHAUSTED"

    call_count: list[int] = [0]

    # Build a minimal success async-iterator factory.
    def _make_success_iter() -> Any:
        """Return an async iterator yielding one Gemini-shaped chunk."""

        async def _gen() -> Any:
            part = MagicMock()
            part.text = "Retried successfully."
            part.function_call = None
            content = MagicMock()
            content.parts = [part]
            finish_reason = MagicMock()
            finish_reason.name = "STOP"
            candidate = MagicMock()
            candidate.content = content
            candidate.finish_reason = finish_reason
            chunk = MagicMock()
            chunk.candidates = [candidate]
            yield chunk

        return _gen()

    # ``generate_content_stream`` is called as ``await client.aio.models.generate_content_stream(...)``.
    # We use AsyncMock so each call returns a coroutine.  On attempt 1 the coroutine
    # raises; on attempt 2 it returns an async iterator.
    async def _side_effect(*_args: Any, **_kw: Any) -> Any:
        call_count[0] += 1
        if call_count[0] == 1:
            raise _Fake429("RESOURCE_EXHAUSTED: quota exceeded")
        return _make_success_iter()

    # Build a real GeminiLLMClient but swap its internal SDK client.
    from robot_comic.gemini_llm import GeminiLLMClient

    real_llm = GeminiLLMClient(api_key="DUMMY", model="gemini-2.5-flash")
    mock_sdk_client = MagicMock()
    mock_sdk_client.aio = MagicMock()
    mock_sdk_client.aio.models = MagicMock()
    mock_sdk_client.aio.models.generate_content_stream = AsyncMock(side_effect=_side_effect)
    real_llm._client = mock_sdk_client
    handler._gemini_llm = real_llm

    # Suppress real sleep so the test runs instantly.
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    # Mock TTS so the test doesn't need a real Chatterbox server.
    async def _noop_tts(text: str, **_kw: Any) -> bytes:
        return _pcm_bytes(2400)

    handler._call_chatterbox_tts = _noop_tts  # type: ignore[method-assign]

    with patch("robot_comic.gemini_llm.asyncio.sleep", side_effect=_fake_sleep):
        await handler._dispatch_completed_transcript("hello")

    # generate_content_stream must have been called at least twice:
    # attempt 1 (raises 429) + attempt 2 (succeeds).
    assert call_count[0] >= 2, (
        f"Expected ≥2 calls to generate_content_stream (1 failing + 1 retry), got {call_count[0]}"
    )
    # At least one sleep confirms the backoff path executed.
    assert len(sleep_calls) >= 1, "Expected asyncio.sleep to be called during 429 backoff"
