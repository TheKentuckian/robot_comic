# Gemini TTS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `gemini_tts` as a selectable output backend under the existing `local_stt` (Custom audio path) flow, composing Moonshine STT input with Gemini 3.1 Flash TTS voice output.

**Architecture:** `GeminiTTSResponseHandler` extends `ConversationHandler` directly (no WebSocket), implementing a request/response cycle: Gemini Flash text model for reasoning + tools, Gemini TTS for voice synthesis. `LocalSTTGeminiTTSHandler` composes `LocalSTTInputMixin` with this new handler via Python MRO — the same mixin pattern used by the two existing LocalSTT handlers. A small surgical refactor to `LocalSTTInputMixin._handle_local_stt_event` extracts `_dispatch_completed_transcript()` as an overrideable hook, enabling the new handler to redirect completed transcripts to Gemini instead of the OpenAI WebSocket.

**Tech Stack:** Python 3.12, `google-genai` SDK (`genai.Client`, `types.Content`, `types.Part`, `types.GenerateContentConfig`, `types.SpeechConfig`, `types.VoiceConfig`, `types.PrebuiltVoiceConfig`, `types.FunctionResponse`), `asyncio`, `numpy`, `fastrtc.AdditionalOutputs`, `fastrtc.wait_for_item`.

**Spec:** `docs/superpowers/specs/2026-05-06-gemini-tts-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/robot_comic/config.py` | Add `GEMINI_TTS_OUTPUT`, `GEMINI_TTS_AVAILABLE_VOICES`, `GEMINI_TTS_DEFAULT_VOICE`; expand `LOCAL_STT_RESPONSE_BACKEND_CHOICES` |
| Modify | `src/robot_comic/local_stt_realtime.py` | Extract `_dispatch_completed_transcript()` hook from `_handle_local_stt_event` |
| Create | `src/robot_comic/gemini_tts.py` | `GeminiTTSResponseHandler` + `LocalSTTGeminiTTSHandler` |
| Modify | `src/robot_comic/main.py` | Dispatch `gemini_tts` response backend in the `local_stt` branch |
| Modify | `src/robot_comic/console.py` | Credential check + persistence for `gemini_tts` output |
| Modify | `src/robot_comic/static/index.html` | Enable the disabled Gemini TTS output card |
| Modify | `src/robot_comic/static/main.js` | Recognize `gemini_tts` in output selection, journey map, credential logic |
| Modify | `profiles/don_rickles/instructions.txt` | Add TTS tag guidance section |
| Create | `tests/test_gemini_tts_handler.py` | Unit tests for `GeminiTTSResponseHandler` |
| Create | `tests/test_local_stt_dispatch_hook.py` | Regression tests for the mixin refactor |

---

## Task 1: Config additions

**Files:**
- Modify: `src/robot_comic/config.py`

- [ ] **Step 1: Add constants to `config.py`**

Open `src/robot_comic/config.py`. After the existing `HF_BACKEND`, `GEMINI_BACKEND`, `OPENAI_BACKEND`, `LOCAL_STT_BACKEND` block (around line 91), add:

```python
GEMINI_TTS_OUTPUT = "gemini_tts"
```

After `GEMINI_AVAILABLE_VOICES` (around line 86), add:

```python
# Voices supported by the Gemini TTS API (gemini-3.1-flash-tts-preview)
GEMINI_TTS_AVAILABLE_VOICES: list[str] = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]
GEMINI_TTS_DEFAULT_VOICE = "Algenib"
```

Update `LOCAL_STT_RESPONSE_BACKEND_CHOICES` (around line 144):

```python
LOCAL_STT_RESPONSE_BACKEND_CHOICES = (OPENAI_BACKEND, HF_BACKEND, GEMINI_TTS_OUTPUT)
```

- [ ] **Step 2: Verify normalization still works**

Run: `python -c "from robot_comic.config import LOCAL_STT_RESPONSE_BACKEND_CHOICES, GEMINI_TTS_OUTPUT; print(GEMINI_TTS_OUTPUT in LOCAL_STT_RESPONSE_BACKEND_CHOICES)"`

Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/robot_comic/config.py
git commit -m "feat: add GEMINI_TTS_OUTPUT backend constant and 30-voice TTS voice list"
```

---

## Task 2: Mixin refactor — extract `_dispatch_completed_transcript()`

**Files:**
- Modify: `src/robot_comic/local_stt_realtime.py`
- Create: `tests/test_local_stt_dispatch_hook.py`

- [ ] **Step 1: Write regression tests first**

Create `tests/test_local_stt_dispatch_hook.py`:

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.local_stt_realtime import (
    LocalSTTOpenAIRealtimeHandler,
    LocalSTTRealtimeHandler,
)


@pytest.mark.asyncio
async def test_openai_handler_dispatch_calls_connection() -> None:
    """_dispatch_completed_transcript on the OpenAI handler sends to the realtime WebSocket."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)

    item = SimpleNamespace(create=AsyncMock())
    handler.connection = SimpleNamespace(conversation=SimpleNamespace(item=item))
    handler._safe_response_create = AsyncMock()  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("hello there")

    item.create.assert_awaited_once()
    sent = item.create.await_args.kwargs["item"]
    assert sent["role"] == "user"
    assert sent["content"][0]["text"] == "hello there"
    handler._safe_response_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_handler_dispatch_no_op_when_disconnected() -> None:
    """_dispatch_completed_transcript is silent when connection is None."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)
    handler.connection = None

    # Should not raise
    await handler._dispatch_completed_transcript("hello there")


@pytest.mark.asyncio
async def test_handle_local_stt_event_delegates_to_dispatch() -> None:
    """_handle_local_stt_event calls _dispatch_completed_transcript for completed events."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)
    handler._dispatch_completed_transcript = AsyncMock()  # type: ignore[method-assign]

    await handler._handle_local_stt_event("completed", "test transcript")

    handler._dispatch_completed_transcript.assert_awaited_once_with("test transcript")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_local_stt_dispatch_hook.py -v
```

Expected: FAIL — `_dispatch_completed_transcript` not defined yet.

- [ ] **Step 3: Refactor `local_stt_realtime.py`**

In `src/robot_comic/local_stt_realtime.py`, find `_handle_local_stt_event` (around line 174). Replace the block that starts with `await self.output_queue.put(...)` and ends with `await self._safe_response_create(...)` with a delegation call:

```python
    async def _handle_local_stt_event(self, kind: str, text: str) -> None:
        """Handle local STT lifecycle events inside the asyncio loop."""
        transcript = (text or "").strip()
        if kind == "started":
            self._mark_activity("local_stt_speech_started")  # type: ignore[attr-defined]
            self._turn_user_done_at = None
            self._turn_response_created_at = None
            self._turn_first_audio_at = None
            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            return

        if kind == "partial":
            if transcript:
                self._mark_activity("local_stt_partial")  # type: ignore[attr-defined]
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
            return

        if kind != "completed" or not transcript:
            return

        now = time.perf_counter()
        if transcript == self._last_completed_transcript and now - self._last_completed_at < 0.75:
            logger.debug("Ignoring duplicate local STT completion: %s", transcript)
            return
        self._last_completed_transcript = transcript
        self._last_completed_at = now

        self._mark_activity("local_stt_completed")  # type: ignore[attr-defined]
        self.deps.movement_manager.set_listening(False)
        self._turn_user_done_at = now  # type: ignore[attr-defined]
        self._turn_response_created_at = None  # type: ignore[attr-defined]
        self._turn_first_audio_at = None  # type: ignore[attr-defined]

        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        await self._dispatch_completed_transcript(transcript)

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """Send a completed transcript to the realtime response backend.

        Override in subclasses to redirect to a different response backend.
        """
        if not self.connection:  # type: ignore[attr-defined]
            logger.debug("Local STT transcript ready but realtime connection is not connected")
            return

        await self.connection.conversation.item.create(  # type: ignore[attr-defined]
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": transcript}],
            },
        )
        await self._safe_response_create(  # type: ignore[attr-defined]
            response=RealtimeResponseCreateParamsParam(
                instructions="Answer the user's transcribed speech naturally and concisely in audio.",
            ),
        )
```

- [ ] **Step 4: Run all local STT tests**

```bash
pytest tests/test_local_stt_realtime.py tests/test_local_stt_dispatch_hook.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/robot_comic/local_stt_realtime.py tests/test_local_stt_dispatch_hook.py
git commit -m "refactor: extract _dispatch_completed_transcript() hook from LocalSTTInputMixin"
```

---

## Task 3: `GeminiTTSResponseHandler` — core handler

**Files:**
- Create: `src/robot_comic/gemini_tts.py`
- Create: `tests/test_gemini_tts_handler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_gemini_tts_handler.py`:

```python
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _make_handler():
    from robot_comic.gemini_tts import GeminiTTSResponseHandler
    deps = _make_deps()
    handler = GeminiTTSResponseHandler(deps)
    handler._client = MagicMock()
    return handler


@pytest.mark.asyncio
async def test_conversation_history_accumulates() -> None:
    """Each transcript + response is appended to history in the correct Gemini format."""
    handler = _make_handler()

    async def fake_llm() -> str:
        return "You look like you comb your hair with a pork chop."

    async def fake_tts(text: str) -> bytes:
        return b"\x00\x01" * 2400  # 2400 samples of silence

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("tell me a joke")

    assert len(handler._conversation_history) == 2
    assert handler._conversation_history[0]["role"] == "user"
    assert handler._conversation_history[0]["parts"][0]["text"] == "tell me a joke"
    assert handler._conversation_history[1]["role"] == "model"
    assert "pork chop" in handler._conversation_history[1]["parts"][0]["text"]


@pytest.mark.asyncio
async def test_pcm_bytes_are_chunked_and_queued() -> None:
    """TTS PCM output is split into ~2400-sample frames and pushed to output_queue."""
    handler = _make_handler()

    # 7200 samples = 3 frames of 2400
    raw_samples = np.zeros(7200, dtype=np.int16)
    pcm_bytes = raw_samples.tobytes()

    async def fake_llm() -> str:
        return "Hockey puck."

    async def fake_tts(text: str) -> bytes:
        return pcm_bytes

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("say something")

    audio_frames = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, tuple):
            audio_frames.append(item)

    assert len(audio_frames) == 3
    sample_rate, samples = audio_frames[0]
    assert sample_rate == 24000
    assert len(samples) == 2400


@pytest.mark.asyncio
async def test_tts_retry_on_failure() -> None:
    """TTS is retried up to 3 times on exception before pushing an error message."""
    handler = _make_handler()

    async def fake_llm() -> str:
        return "Beautiful."

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]

    call_count = 0

    async def failing_tts(text: str) -> bytes:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("500 Internal Server Error")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        handler._call_tts_with_retry = failing_tts  # type: ignore[method-assign]
        # We patch _call_tts_with_retry directly to simulate all retries failing
        # so we need to test _call_tts_with_retry directly instead
        pass

    # Test _call_tts_with_retry directly
    handler2 = _make_handler()
    attempt_count = 0

    async def always_fails(text: str) -> None:
        raise RuntimeError("boom")

    original_call = handler2._call_tts_with_retry

    # Patch the internal genai call
    with patch.object(handler2, "_client") as mock_client:
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("500"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await handler2._call_tts_with_retry("Hello")

    assert result is None


@pytest.mark.asyncio
async def test_tts_failure_pushes_error_output() -> None:
    """When TTS fails after all retries, an error AdditionalOutputs is pushed to the queue."""
    handler = _make_handler()

    async def fake_llm() -> str:
        return "Beautiful."

    async def failing_tts(text: str) -> None:
        return None  # simulate all retries exhausted

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = failing_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("say something")

    items = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())

    error_items = [i for i in items if isinstance(i, AdditionalOutputs) and "error" in str(i.args).lower()]
    assert len(error_items) >= 1


@pytest.mark.asyncio
async def test_voice_override() -> None:
    """get_current_voice returns the override when set, otherwise the default."""
    from robot_comic.gemini_tts import GeminiTTSResponseHandler
    from robot_comic.config import GEMINI_TTS_DEFAULT_VOICE

    handler = GeminiTTSResponseHandler(_make_deps())
    assert handler.get_current_voice() == GEMINI_TTS_DEFAULT_VOICE

    handler._voice_override = "Puck"
    assert handler.get_current_voice() == "Puck"


@pytest.mark.asyncio
async def test_apply_personality_clears_history() -> None:
    """apply_personality resets conversation history."""
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    handler = GeminiTTSResponseHandler(_make_deps())
    handler._conversation_history = [{"role": "user", "parts": [{"text": "hi"}]}]

    with patch("robot_comic.gemini_tts.set_custom_profile"):
        await handler.apply_personality("don_rickles")

    assert handler._conversation_history == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_gemini_tts_handler.py -v
```

Expected: FAIL — `robot_comic.gemini_tts` does not exist.

- [ ] **Step 3: Create `src/robot_comic/gemini_tts.py`**

```python
"""Gemini TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls the
Gemini Flash text model for reasoning and tool execution, then synthesises
audio with gemini-3.1-flash-tts-preview (voice: Algenib by default).

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
"""

import json
import base64
import asyncio
import logging
from typing import Any, Optional

import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item
from google import genai
from google.genai import types

from robot_comic.config import (
    GEMINI_TTS_AVAILABLE_VOICES,
    GEMINI_TTS_DEFAULT_VOICE,
    config,
)
from robot_comic.conversation_handler import AudioFrame, ConversationHandler
from robot_comic.gemini_live import _openai_tool_specs_to_gemini
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.prompts import get_session_instructions
from robot_comic.tools.core_tools import ToolDependencies, dispatch_tool_call, get_active_tool_specs

logger = logging.getLogger(__name__)

GEMINI_TTS_LLM_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_TTS_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz
_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5
_LLM_MAX_TOOL_ROUNDS = 5


class GeminiTTSResponseHandler(ConversationHandler):
    """Request/response handler: Gemini Flash text model + Gemini TTS voice output."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override: str | None = startup_voice
        self._client: genai.Client | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._clear_queue: Any = None

        # Attributes set by LocalSTTInputMixin (declared here to satisfy type checker)
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    def _mark_activity(self, label: str) -> None:
        """No-op activity marker (no cost tracking in this handler)."""
        logger.debug("Activity: %s", label)

    def copy(self) -> "GeminiTTSResponseHandler":
        return GeminiTTSResponseHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        """Initialise the Gemini client. Called via MRO by LocalSTTInputMixin."""
        api_key = config.GEMINI_API_KEY or "DUMMY"
        self._client = genai.Client(api_key=api_key)
        logger.info(
            "GeminiTTS handler initialised: llm=%s tts=%s voice=%s",
            GEMINI_TTS_LLM_MODEL,
            GEMINI_TTS_MODEL,
            self.get_current_voice(),
        )

    async def start_up(self) -> None:
        """Initialise via MRO and block until shutdown() is called."""
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        await self._stop_event.wait()

    async def shutdown(self) -> None:
        """Signal start_up to return and drain the output queue."""
        self._stop_event.set()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive(self, frame: AudioFrame) -> None:
        """No-op: audio input is handled by LocalSTTInputMixin.receive()."""

    async def emit(self) -> Any:
        """Yield the next audio frame or status message from the output queue."""
        return await wait_for_item(self.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality profile and reset conversation history."""
        try:
            from robot_comic.config import set_custom_profile
            set_custom_profile(profile)
            self._conversation_history.clear()
            return f"Applied personality {profile!r}. Conversation history reset."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        return list(GEMINI_TTS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        return self._voice_override or GEMINI_TTS_DEFAULT_VOICE

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # Response cycle                                                       #
    # ------------------------------------------------------------------ #

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """Gemini-native response cycle: LLM → tools → TTS → audio frames."""
        self._conversation_history.append(
            {"role": "user", "parts": [{"text": transcript}]}
        )

        try:
            response_text = await self._run_llm_with_tools()
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return

        self._conversation_history.append(
            {"role": "model", "parts": [{"text": response_text}]}
        )
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": response_text})
        )

        pcm_bytes = await self._call_tts_with_retry(response_text)
        if pcm_bytes is None:
            await self.output_queue.put(
                AdditionalOutputs(
                    {"role": "assistant", "content": "[TTS error — could not generate audio]"}
                )
            )
            return

        for frame in self._pcm_to_frames(pcm_bytes):
            await self.output_queue.put((GEMINI_TTS_OUTPUT_SAMPLE_RATE, frame))

    async def _run_llm_with_tools(self) -> str:
        """Call Gemini Flash with conversation history, handling tool round-trips."""
        assert self._client is not None, "Client not initialised"

        tool_specs = get_active_tool_specs(self.deps)
        function_declarations = _openai_tool_specs_to_gemini(tool_specs)
        tools_config = (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )
        gen_config = types.GenerateContentConfig(
            system_instruction=get_session_instructions(),
            tools=tools_config,  # type: ignore[arg-type]
        )

        history: list[Any] = list(self._conversation_history)

        for _ in range(_LLM_MAX_TOOL_ROUNDS):
            response = await self._client.aio.models.generate_content(
                model=GEMINI_TTS_LLM_MODEL,
                contents=history,
                config=gen_config,
            )

            candidate = response.candidates[0]
            function_calls = [
                p.function_call
                for p in candidate.content.parts
                if p.function_call is not None
            ]

            if not function_calls:
                return "".join(
                    p.text for p in candidate.content.parts if p.text
                ).strip()

            # Append model's function-call turn to history
            history.append(candidate.content)

            # Execute tools and collect responses
            response_parts: list[types.Part] = []
            for fc in function_calls:
                logger.info("GeminiTTS tool call: %s args=%s", fc.name, dict(fc.args))
                try:
                    result = await dispatch_tool_call(
                        fc.name, json.dumps(dict(fc.args)), self.deps
                    )
                except Exception as exc:
                    result = {"error": str(exc)}

                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name, response=result
                        )
                    )
                )
                await self.output_queue.put(
                    AdditionalOutputs(
                        {"role": "assistant", "content": f"🛠️ Used tool {fc.name}"}
                    )
                )

            history.append(types.Content(role="user", parts=response_parts))

        return "[Response generation reached tool call limit]"

    async def _call_tts_with_retry(self, text: str) -> bytes | None:
        """Call Gemini TTS, retrying up to 3 times on transient errors."""
        assert self._client is not None, "Client not initialised"

        tts_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.get_current_voice()
                    )
                )
            ),
        )

        for attempt in range(_TTS_MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=GEMINI_TTS_MODEL,
                    contents=text,
                    config=tts_config,
                )
                data = response.candidates[0].content.parts[0].inline_data.data
                return base64.b64decode(data) if isinstance(data, str) else bytes(data)
            except Exception as exc:
                logger.warning(
                    "TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc
                )
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_DELAY)

        return None

    @staticmethod
    def _pcm_to_frames(pcm_bytes: bytes) -> list[np.ndarray]:
        """Split raw 16-bit PCM bytes into ~100 ms numpy frames."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return [
            audio[i: i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i: i + _CHUNK_SAMPLES]) > 0
        ]


class LocalSTTGeminiTTSHandler(LocalSTTInputMixin, GeminiTTSResponseHandler):
    """Moonshine STT input + Gemini 3.1 Flash TTS voice output."""
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_gemini_tts_handler.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/robot_comic/gemini_tts.py tests/test_gemini_tts_handler.py
git commit -m "feat: add GeminiTTSResponseHandler and LocalSTTGeminiTTSHandler"
```

---

## Task 4: Wire into `main.py`

**Files:**
- Modify: `src/robot_comic/main.py`

- [ ] **Step 1: Update the `local_stt` dispatch block**

In `src/robot_comic/main.py`, find the `elif config.BACKEND_PROVIDER == LOCAL_STT_BACKEND:` block (around line 213). Replace it with:

```python
    elif config.BACKEND_PROVIDER == LOCAL_STT_BACKEND:
        from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler
        from robot_comic.local_stt_realtime import (
            LocalSTTHuggingFaceRealtimeHandler,
            LocalSTTOpenAIRealtimeHandler,
        )

        local_stt_response_backend = getattr(config, "LOCAL_STT_RESPONSE_BACKEND", OPENAI_BACKEND)
        logger.info(
            "Using %s via local STT input and %s response audio",
            get_backend_label(config.BACKEND_PROVIDER),
            get_backend_label(local_stt_response_backend) if local_stt_response_backend != GEMINI_TTS_OUTPUT else "Gemini TTS",
        )

        if local_stt_response_backend == HF_BACKEND:
            handler_class = LocalSTTHuggingFaceRealtimeHandler
        elif local_stt_response_backend == GEMINI_TTS_OUTPUT:
            handler_class = LocalSTTGeminiTTSHandler
        else:
            handler_class = LocalSTTOpenAIRealtimeHandler

        handler = handler_class(
            deps,
            gradio_mode=args.gradio,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )  # type: ignore[assignment]
```

Also add `GEMINI_TTS_OUTPUT` to the imports from `robot_comic.config` at the top of the function (around line 49):

```python
    from robot_comic.config import (
        HF_BACKEND,
        GEMINI_BACKEND,
        OPENAI_BACKEND,
        LOCAL_STT_BACKEND,
        GEMINI_TTS_OUTPUT,      # ← add this
        HF_LOCAL_CONNECTION_MODE,
        config,
        is_gemini_model,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
```

- [ ] **Step 2: Smoke-check import**

```bash
python -c "from robot_comic.main import run; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/robot_comic/main.py
git commit -m "feat: dispatch LocalSTTGeminiTTSHandler when LOCAL_STT_RESPONSE_BACKEND=gemini_tts"
```

---

## Task 5: Wire into `console.py`

**Files:**
- Modify: `src/robot_comic/console.py`

- [ ] **Step 1: Add import**

In `src/robot_comic/console.py`, find the `from robot_comic.config import (` block (around line 27). Add `GEMINI_TTS_OUTPUT` to it:

```python
from robot_comic.config import (
    HF_BACKEND,
    GEMINI_BACKEND,
    LOCKED_PROFILE,
    OPENAI_BACKEND,
    LOCAL_STT_BACKEND,
    GEMINI_TTS_OUTPUT,          # ← add this
    LOCAL_STT_MODEL_ENV,
    ...
)
```

- [ ] **Step 2: Update `_has_required_key()`**

Find `_has_required_key` (around line 181). Update the `LOCAL_STT_BACKEND` branch:

```python
    def _has_required_key(self, backend: str) -> bool:
        if backend == GEMINI_BACKEND:
            return self._has_key(config.GEMINI_API_KEY)
        if backend == HF_BACKEND:
            return has_hf_realtime_target()
        if backend == LOCAL_STT_BACKEND:
            response_backend = getattr(config, "LOCAL_STT_RESPONSE_BACKEND", OPENAI_BACKEND)
            if response_backend == HF_BACKEND:
                return has_hf_realtime_target()
            if response_backend == GEMINI_TTS_OUTPUT:
                return self._has_key(config.GEMINI_API_KEY)
            return self._has_key(config.OPENAI_API_KEY)
        return self._has_key(config.OPENAI_API_KEY)
```

- [ ] **Step 3: Update `_requirement_name()`**

Find `_requirement_name` (around line 193). Update the `LOCAL_STT_BACKEND` branch:

```python
    @staticmethod
    def _requirement_name(backend: str) -> str:
        if backend == GEMINI_BACKEND:
            return "GEMINI_API_KEY"
        if backend == HF_BACKEND:
            return HF_REALTIME_WS_URL_ENV
        if backend == LOCAL_STT_BACKEND:
            response_backend = getattr(config, "LOCAL_STT_RESPONSE_BACKEND", OPENAI_BACKEND)
            if response_backend == HF_BACKEND:
                return HF_REALTIME_WS_URL_ENV
            if response_backend == GEMINI_TTS_OUTPUT:
                return "GEMINI_API_KEY"
            return "OPENAI_API_KEY"
        return "OPENAI_API_KEY"
```

- [ ] **Step 4: Update `_set_backend()` — validation**

In `_set_backend()` (around line 495), find the block that validates `local_stt_response_backend`. After the existing `if backend == LOCAL_STT_BACKEND and local_stt_response_backend == OPENAI_BACKEND and not api_key ...` check, add a Gemini key check:

```python
            if (
                backend == LOCAL_STT_BACKEND
                and local_stt_response_backend == GEMINI_TTS_OUTPUT
                and not api_key
                and not self._has_key(config.GEMINI_API_KEY)
            ):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
```

- [ ] **Step 5: Update `_set_backend()` — key persistence**

In the same function, find where keys are persisted (around line 517):

```python
            if backend == OPENAI_BACKEND and api_key:
                self._persist_api_key(api_key)
            if backend == LOCAL_STT_BACKEND and local_stt_response_backend == OPENAI_BACKEND and api_key:
                self._persist_api_key(api_key)
            if backend == GEMINI_BACKEND and api_key:
                self._persist_gemini_api_key(api_key)
```

Add after the Gemini line:

```python
            if backend == LOCAL_STT_BACKEND and local_stt_response_backend == GEMINI_TTS_OUTPUT and api_key:
                self._persist_gemini_api_key(api_key)
```

- [ ] **Step 6: Update the HF connection block condition**

Find the line that reads `if backend == HF_BACKEND or (backend == LOCAL_STT_BACKEND and local_stt_response_backend == HF_BACKEND):` (around line 557). No change needed here — this already handles the HF case for local_stt.

- [ ] **Step 7: Verify console imports and run tests**

```bash
python -c "from robot_comic.console import LocalStream; print('ok')"
pytest tests/test_console.py -v
```

Expected: `ok` and all console tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/robot_comic/console.py
git commit -m "feat: credential check and persistence for local_stt + gemini_tts output"
```

---

## Task 6: Admin UI — `index.html`

**Files:**
- Modify: `src/robot_comic/static/index.html`

- [ ] **Step 1: Enable the Gemini TTS output card**

Find this block in `src/robot_comic/static/index.html` (around line 173):

```html
            <label class="output-choice is-disabled" data-output-card="gemini_tts">
              <input type="radio" name="local-stt-output" value="gemini_tts" disabled />
              <span>
                <strong>Gemini Flash 3.1 TTS</strong>
                <small>Placeholder for a future text-to-speech output route.</small>
              </span>
            </label>
```

Replace with:

```html
            <label class="output-choice" data-output-card="gemini_tts">
              <input type="radio" name="local-stt-output" value="gemini_tts" />
              <span>
                <strong>Gemini Flash 3.1 TTS</strong>
                <small>On-device Moonshine STT → Gemini text response → Gemini 3.1 Flash TTS. Requires <code>GEMINI_API_KEY</code>.</small>
              </span>
            </label>
```

- [ ] **Step 2: Verify HTML is valid**

Open `src/robot_comic/static/index.html` in a browser or run:

```bash
python -c "
from pathlib import Path
html = Path('src/robot_comic/static/index.html').read_text()
assert 'is-disabled' not in html or html.count('is-disabled') == 0, 'disabled class still present'
assert 'disabled' not in html.split('gemini_tts')[1].split('</label>')[0], 'disabled attr still present'
print('ok')
"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/robot_comic/static/index.html
git commit -m "feat: enable Gemini Flash 3.1 TTS output card in admin UI"
```

---

## Task 7: Admin UI — `main.js`

**Files:**
- Modify: `src/robot_comic/static/main.js`

- [ ] **Step 1: Add `GEMINI_TTS_OUTPUT` constant**

At the top of `src/robot_comic/static/main.js`, after the existing backend constants (around line 4):

```javascript
const GEMINI_TTS_OUTPUT = "gemini_tts";
```

- [ ] **Step 2: Update `setSelectedLocalSTTOutput()`**

Find `setSelectedLocalSTTOutput` (around line 515). Replace:

```javascript
  function setSelectedLocalSTTOutput(outputBackend) {
    const normalized = outputBackend === HF_BACKEND ? HF_BACKEND : OPENAI_BACKEND;
```

With:

```javascript
  function setSelectedLocalSTTOutput(outputBackend) {
    const normalized = outputBackend === HF_BACKEND
      ? HF_BACKEND
      : outputBackend === GEMINI_TTS_OUTPUT
        ? GEMINI_TTS_OUTPUT
        : OPENAI_BACKEND;
```

- [ ] **Step 3: Update `journeyMeta()` for gemini_tts**

Find `journeyMeta()` (around line 129). Update the `if (backend === LOCAL_STT_BACKEND)` branch:

```javascript
  if (backend === LOCAL_STT_BACKEND) {
    if (outputBackend === HF_BACKEND) {
      meta.brainLabel = "Hugging Face response backend";
      meta.outputLabel = "Hugging Face voice";
      meta.outputCopy = "Speech comes back through the configured Hugging Face endpoint.";
    } else if (outputBackend === GEMINI_TTS_OUTPUT) {
      meta.brainLabel = "Gemini Flash response backend";
      meta.outputLabel = "Gemini Flash 3.1 TTS";
      meta.outputCopy = "Speech comes back through Gemini 3.1 Flash TTS with the Algenib voice.";
    } else {
      meta.brainLabel = "OpenAI response backend";
      meta.outputLabel = "OpenAI voice";
      meta.outputCopy = "Speech comes back through OpenAI text-in, audio-out realtime.";
    }
  }
```

- [ ] **Step 4: Update credential panel logic for gemini_tts**

Find `renderCredentialPanels` (around line 536). Update the local STT credential variables:

```javascript
    const localSttUsesHF = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === HF_BACKEND;
    const localSttUsesGeminiTTS = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === GEMINI_TTS_OUTPUT;
    const localSttUsesOpenAI = selectedBackend === LOCAL_STT_BACKEND && !localSttUsesHF && !localSttUsesGeminiTTS;
    const canProceedWithSelectedBackend = localSttUsesHF
      ? backendCanProceed(status, HF_BACKEND)
      : localSttUsesGeminiTTS
        ? backendCanProceed(status, GEMINI_BACKEND)
        : localSttUsesOpenAI
          ? backendCanProceed(status, OPENAI_BACKEND)
          : backendCanProceed(status, selectedBackend);
    const usesApiKeyForm = selectedBackend === OPENAI_BACKEND || selectedBackend === GEMINI_BACKEND || localSttUsesOpenAI || localSttUsesGeminiTTS;
```

Also update `apiKeyLabel` to reflect the correct key name when Gemini TTS is selected. Find where `apiKeyLabel.textContent = meta.inputLabel;` is set and add an override after it:

```javascript
    apiKeyLabel.textContent = meta.inputLabel;
    if (localSttUsesGeminiTTS) {
      apiKeyLabel.textContent = "GEMINI_API_KEY";
      input.placeholder = "AIza...";
    }
```

- [ ] **Step 5: Manual smoke-test**

Start the app in headless mode with a settings server and open the admin UI. Select "Custom audio path", then select "Gemini Flash 3.1 TTS" as output. Verify:
- The journey map shows "Gemini Flash response backend" and "Gemini Flash 3.1 TTS"
- The credential form shows "GEMINI_API_KEY" label
- Switching back to OpenAI output shows "OPENAI_API_KEY"

If the live server is not available, skip to the next step.

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/static/main.js
git commit -m "feat: recognize gemini_tts in admin UI output selection and journey map"
```

---

## Task 8: Don Rickles profile — TTS tag guidance

**Files:**
- Modify: `profiles/don_rickles/instructions.txt`

- [ ] **Step 1: Add TTS tag section**

Open `profiles/don_rickles/instructions.txt`. At the end of the file, append:

```
## GEMINI TTS DELIVERY TAGS

When running with Gemini TTS, embed inline delivery tags directly in your responses to shape the voice output. Tags go inside square brackets and must be separated by spoken text — never place two tags adjacent to each other.

**Use these freely:**
- `[fast]` — rapid-fire delivery; use for insult volleys and quick callbacks
- `[annoyance]` — for the peak of contempt ("Look at this guy...")
- `[aggression]` — for the sharp edge of a well-landed roast line
- `[amusement]` — immediately after a particularly sharp line (Rickles loved his own jokes)
- `[enthusiasm]` — for the fake warmth before a devastating pivot

**Pacing:**
- `[short pause]` — a beat after landing a punch, before the next question
- `[slow]` — for the set-up of a long roast; contrast with the [fast] payoff

**Rule:** One tag per sentence maximum. Less is more. Let the persona carry the delivery; tags only sharpen the peaks.

**Example response:**
"[annoyance] Oh, look at you. [short pause] [fast] You comb your hair with a pork chop, you hockey puck! [amusement] Beautiful. [short pause] Now tell me — where are you from? [fast] And don't say New Jersey, I can only take so much."
```

- [ ] **Step 2: Verify file reads cleanly**

```bash
python -c "
from robot_comic.prompts import get_session_instructions
from robot_comic.config import set_custom_profile
set_custom_profile('don_rickles')
instructions = get_session_instructions()
assert 'GEMINI TTS' in instructions, 'TTS section not found'
print('ok')
"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add profiles/don_rickles/instructions.txt
git commit -m "feat: add Gemini TTS delivery tag guidance to Don Rickles instructions"
```

---

## Task 9: Full test run and integration check

**Files:** None new.

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS (no regressions).

- [ ] **Step 2: Verify config round-trip**

```bash
python -c "
import os
os.environ['BACKEND_PROVIDER'] = 'local_stt'
os.environ['LOCAL_STT_RESPONSE_BACKEND'] = 'gemini_tts'
from robot_comic.config import refresh_runtime_config_from_env, config
refresh_runtime_config_from_env()
assert config.LOCAL_STT_RESPONSE_BACKEND == 'gemini_tts', config.LOCAL_STT_RESPONSE_BACKEND
print('config ok')
"
```

Expected: `config ok`

- [ ] **Step 3: Verify handler import chain**

```bash
python -c "
from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler
from robot_comic.tools.core_tools import ToolDependencies
from unittest.mock import MagicMock
deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
h = LocalSTTGeminiTTSHandler(deps)
print('MRO:', [c.__name__ for c in type(h).__mro__[:5]])
print('handler ok')
"
```

Expected output contains: `LocalSTTGeminiTTSHandler`, `LocalSTTInputMixin`, `GeminiTTSResponseHandler`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Gemini TTS integration complete — Moonshine STT + Gemini Flash TTS voice output"
```
