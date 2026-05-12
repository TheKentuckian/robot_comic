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
from google import genai
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from google.genai import types

from robot_comic.config import (
    GEMINI_AVAILABLE_VOICES,
    GEMINI_TTS_DEFAULT_VOICE,
    GEMINI_TTS_AVAILABLE_VOICES,
    config,
    set_custom_profile,
)


# Voices shared with Gemini Live may have been persisted for that backend.
# Only restore startup_voice if it is TTS-exclusive (not in the Live voice list).
_TTS_EXCLUSIVE_VOICES: frozenset[str] = frozenset(GEMINI_TTS_AVAILABLE_VOICES) - frozenset(GEMINI_AVAILABLE_VOICES)
from robot_comic.prompts import get_session_instructions
from robot_comic.gemini_live import _openai_tool_specs_to_gemini
from robot_comic.tools.core_tools import ToolDependencies, dispatch_tool_call, get_active_tool_specs
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)

GEMINI_TTS_LLM_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_TTS_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz
_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5
_LLM_MAX_RETRIES = 4
_LLM_RETRY_BASE_DELAY = 1.0
_LLM_MAX_TOOL_ROUNDS = 5


class GeminiTTSResponseHandler(AsyncStreamHandler, ConversationHandler):
    """Request/response handler: Gemini Flash text model + Gemini TTS voice output."""

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.sim_mode = sim_mode
        self.instance_path = instance_path
        # Only restore voices that are TTS-exclusive. Shared Live/TTS voices
        # (Kore, Aoede, etc.) may have been persisted for Gemini Live — ignore them
        # and default to GEMINI_TTS_DEFAULT_VOICE (Algenib) instead.
        self._voice_override: str | None = startup_voice if startup_voice in _TTS_EXCLUSIVE_VOICES else None
        self._client: genai.Client | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Attributes referenced by LocalSTTInputMixin (declared to satisfy type checker and MRO)
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    def _mark_activity(self, label: str) -> None:
        """No-op activity marker (no cost tracking in this handler)."""
        logger.debug("Activity: %s", label)

    def copy(self) -> "GeminiTTSResponseHandler":
        return GeminiTTSResponseHandler(
            self.deps,
            self.sim_mode,
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
        """Initialise credentials and block until shutdown() is called."""
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        asyncio.create_task(self._send_startup_trigger(), name="gemini-tts-startup-trigger")
        await self._stop_event.wait()

    async def _send_startup_trigger(self) -> None:
        """Kick off the opening sequence defined in the profile instructions."""
        await self._dispatch_completed_transcript("[conversation started]")

    async def shutdown(self) -> None:
        """Signal start_up to return and drain the output queue."""
        self._stop_event.set()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive(self, frame: Any) -> None:
        """No-op: audio input is handled by LocalSTTInputMixin.receive()."""

    async def emit(self) -> Any:
        """Yield the next audio frame or status message from the output queue."""
        return await wait_for_item(self.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality profile and reset conversation history."""
        try:
            set_custom_profile(profile)
            self._conversation_history.clear()
            return f"Applied personality {profile!r}. Conversation history reset."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        return list(GEMINI_TTS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        voice = self._voice_override or GEMINI_TTS_DEFAULT_VOICE
        if voice not in GEMINI_TTS_AVAILABLE_VOICES:
            logger.warning("Voice %r is not a valid Gemini TTS voice; falling back to %s", voice, GEMINI_TTS_DEFAULT_VOICE)
            return GEMINI_TTS_DEFAULT_VOICE
        return voice

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

    async def _llm_generate_with_backoff(self, contents: Any, config: Any) -> Any:
        """Call generate_content with exponential backoff on 503/UNAVAILABLE errors."""
        assert self._client is not None, "Client not initialised"
        delay = _LLM_RETRY_BASE_DELAY
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                return await self._client.aio.models.generate_content(
                    model=GEMINI_TTS_LLM_MODEL,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                msg = str(exc)
                is_retryable = "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if not is_retryable or attempt == _LLM_MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "LLM attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1, _LLM_MAX_RETRIES, msg.split("\n")[0], delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

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
            response = await self._llm_generate_with_backoff(history, gen_config)

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
            system_instruction=(
                "Deliver this text at a fast, clipped Brooklyn pace — "
                "rapid-fire on the insults, short crisp pauses only where marked. "
                "Never drawl or over-enunciate. Keep the energy sharp."
            ),
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

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        # LocalSTTInputMixin._dispatch_completed_transcript is OpenAI-specific and
        # shadows GeminiTTSResponseHandler's override due to MRO; route explicitly.
        await GeminiTTSResponseHandler._dispatch_completed_transcript(self, transcript)
