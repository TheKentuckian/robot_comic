"""ElevenLabs TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls the
Gemini Flash text model for reasoning and tool execution, then synthesises
audio with ElevenLabs TTS API.

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
"""

import json
import base64
import asyncio
import logging
from typing import Any, Optional

import numpy as np
import httpx
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from google import genai
from google.genai import types

from robot_comic.config import (
    ELEVENLABS_AVAILABLE_VOICES,
    ELEVENLABS_DEFAULT_VOICE,
    config,
    set_custom_profile,
)
from robot_comic.llama_base import split_sentences
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.tools.core_tools import ToolDependencies, dispatch_tool_call, get_active_tool_specs
from robot_comic.prompts import get_session_instructions
from robot_comic.gemini_live import _openai_tool_specs_to_gemini
from robot_comic.gemini_retry import (
    compute_backoff,
    is_rate_limit_error,
    describe_quota_failure,
    extract_retry_after_seconds,
)
from robot_comic.history_trim import trim_history_in_place
from robot_comic.conversation_handler import ConversationHandler

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_LLM_MODEL = "gemini-2.5-flash"
ELEVENLABS_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz
_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY = 0.5
_LLM_MAX_RETRIES = 4
_LLM_RETRY_BASE_DELAY = 1.0
_LLM_MAX_TOOL_ROUNDS = 5

# Map voice names to ElevenLabs voice IDs
_ELEVENLABS_VOICE_IDS = {
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Antoni": "ErXwobaYp0GwwMsXgNVH",
    "Domi": "AZnzlk1UV0MYJmxZNSD4",
    "Elli": "MF3mGyEYCl7XYWbV7PZT",
    "Gigi": "jsCqWAovK2LkecY7zXl4",
    "Freya": "jKsUlyx0O5BjJQ0XfvjQ",
    "Harry": "SOYHLrjzK2X1ezoeGApW",
    "Liam": "FGKprHBWjP1d6XrNSZaE",
    "Rachel": "21m00Tcm4ijWNoXd58YU",
    "River": "SAz9YHcvj6GT2YYXdXnW",
    "Sam": "2EiwWnXFnvU5JabPnv2n",
}


def load_profile_elevenlabs_config() -> dict[str, str]:
    """Read profiles/<name>/elevenlabs.txt as key=value pairs, if present."""
    profile: str | None = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    if not profile:
        return {}
    try:
        path = config.PROFILES_DIRECTORY / profile / "elevenlabs.txt"
        if not path.exists():
            return {}
        params: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                params[k.strip()] = v.strip()
        return params
    except Exception as exc:
        logger.warning("Could not read elevenlabs.txt for profile %r: %s", profile, exc)
        return {}


class ElevenLabsTTSResponseHandler(AsyncStreamHandler, ConversationHandler):
    """Gemini Flash text model + ElevenLabs TTS voice output."""

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=ELEVENLABS_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.sim_mode = sim_mode
        self.instance_path = instance_path
        self._voice_override: str | None = startup_voice
        self._client: genai.Client | None = None
        self._http: httpx.AsyncClient | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        self._last_tts_rate_limited: bool = False
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Attributes referenced by LocalSTTInputMixin
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    def _mark_activity(self, label: str) -> None:
        """No-op activity marker."""
        logger.debug("Activity: %s", label)

    def copy(self) -> "ElevenLabsTTSResponseHandler":
        return ElevenLabsTTSResponseHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        """Initialise Gemini and HTTP clients."""
        api_key = config.GEMINI_API_KEY or "DUMMY"
        self._client = genai.Client(api_key=api_key)
        self._http = httpx.AsyncClient(timeout=30.0)
        logger.info(
            "ElevenLabsTTS handler initialised: llm=%s voice=%s",
            ELEVENLABS_TTS_LLM_MODEL,
            self.get_current_voice(),
        )

    async def start_up(self) -> None:
        """Initialise credentials and block until shutdown() is called."""
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        asyncio.create_task(self._send_startup_trigger(), name="elevenlabs-tts-startup-trigger")
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
        if self._http:
            await self._http.aclose()

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
        return list(ELEVENLABS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        if self._voice_override:
            return self._voice_override
        config_params = load_profile_elevenlabs_config()
        voice = config_params.get("voice") or ELEVENLABS_DEFAULT_VOICE
        if voice not in ELEVENLABS_AVAILABLE_VOICES:
            logger.warning(
                "Voice %r is not a valid ElevenLabs voice; falling back to %s", voice, ELEVENLABS_DEFAULT_VOICE
            )
            return ELEVENLABS_DEFAULT_VOICE
        return voice

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # Response cycle                                                       #
    # ------------------------------------------------------------------ #

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """ElevenLabs + Gemini response cycle: LLM → tools → TTS → audio frames."""
        self._conversation_history.append({"role": "user", "parts": [{"text": transcript}]})
        trim_history_in_place(self._conversation_history, role_key="role")

        try:
            response_text = await self._run_llm_with_tools()
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return

        self._conversation_history.append({"role": "model", "parts": [{"text": response_text}]})
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))

        sentences = split_sentences(response_text) or [response_text]
        any_audio = False
        for sentence in sentences:
            if not sentence:
                continue
            pcm_bytes = await self._call_elevenlabs_tts(sentence)
            if pcm_bytes is None:
                continue
            for frame in self._pcm_to_frames(pcm_bytes):
                await self.output_queue.put((ELEVENLABS_OUTPUT_SAMPLE_RATE, frame))
            any_audio = True

        if not any_audio:
            if self._last_tts_rate_limited:
                msg = "[ElevenLabs TTS rate-limited; try again later]"
            else:
                msg = "[TTS error — could not generate audio]"
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": msg}))

    async def _llm_generate_with_backoff(self, contents: Any, config: Any) -> Any:
        """Call generate_content with backoff on transient errors and 429s."""
        assert self._client is not None, "Client not initialised"
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                return await self._client.aio.models.generate_content(
                    model=ELEVENLABS_TTS_LLM_MODEL,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                msg = str(exc)
                rate_limited = is_rate_limit_error(exc)
                is_retryable = rate_limited or "503" in msg or "UNAVAILABLE" in msg
                if not is_retryable or attempt == _LLM_MAX_RETRIES - 1:
                    if rate_limited and attempt == _LLM_MAX_RETRIES - 1:
                        logger.error(
                            "Gemini LLM rate-limited (quota=%s) after %d attempts; giving up",
                            describe_quota_failure(exc),
                            _LLM_MAX_RETRIES,
                        )
                    raise
                retry_after = extract_retry_after_seconds(exc) if rate_limited else None
                delay = compute_backoff(attempt, _LLM_RETRY_BASE_DELAY, retry_after)
                if rate_limited:
                    logger.warning(
                        "Gemini LLM 429 (quota=%s, attempt %d/%d); sleeping %.1fs before retry",
                        describe_quota_failure(exc),
                        attempt + 1,
                        _LLM_MAX_RETRIES,
                        delay,
                    )
                else:
                    logger.warning(
                        "Gemini LLM attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1,
                        _LLM_MAX_RETRIES,
                        msg.split("\n")[0],
                        delay,
                    )
                await asyncio.sleep(delay)

    async def _run_llm_with_tools(self) -> str:
        """Call Gemini Flash with conversation history, handling tool round-trips."""
        assert self._client is not None, "Client not initialised"

        tool_specs = get_active_tool_specs(self.deps)
        function_declarations = _openai_tool_specs_to_gemini(tool_specs)
        tools_config = [types.Tool(function_declarations=function_declarations)] if function_declarations else []
        gen_config = types.GenerateContentConfig(
            system_instruction=get_session_instructions(),
            tools=tools_config,  # type: ignore[arg-type]
        )

        history: list[Any] = list(self._conversation_history)

        for _ in range(_LLM_MAX_TOOL_ROUNDS):
            response = await self._llm_generate_with_backoff(history, gen_config)

            candidate = response.candidates[0]
            function_calls = [p.function_call for p in candidate.content.parts if p.function_call is not None]

            if not function_calls:
                return "".join(p.text for p in candidate.content.parts if p.text).strip()

            # Append model's function-call turn to history
            history.append(candidate.content)

            # Execute tools and collect responses
            response_parts: list[types.Part] = []
            for fc in function_calls:
                logger.info("ElevenLabsTTS tool call: %s args=%s", fc.name, dict(fc.args))
                try:
                    result = await dispatch_tool_call(fc.name, json.dumps(dict(fc.args)), self.deps)
                except Exception as exc:
                    result = {"error": str(exc)}

                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(name=fc.name, response=result))
                )
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": f"🛠️ Used tool {fc.name}"})
                )

            history.append(types.Content(role="user", parts=response_parts))

        return "[Response generation reached tool call limit]"

    async def _call_elevenlabs_tts(self, text: str) -> bytes | None:
        """Call ElevenLabs TTS API with retry logic."""
        assert self._http is not None, "Client not initialised"

        api_key = config.ELEVENLABS_API_KEY
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not configured")
            return None

        voice_id = _ELEVENLABS_VOICE_IDS.get(self.get_current_voice())
        if not voice_id:
            logger.error("Could not resolve voice ID for %s", self.get_current_voice())
            return None

        config_params = load_profile_elevenlabs_config()
        stability = float(config_params.get("stability", "0.5"))
        similarity_boost = float(config_params.get("similarity_boost", "0.75"))

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }

        self._last_tts_rate_limited = False
        for attempt in range(_TTS_MAX_RETRIES):
            try:
                response = await self._http.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    self._last_tts_rate_limited = True
                    logger.warning(
                        "ElevenLabs TTS 429 (attempt %d/%d); sleeping %.1fs before retry",
                        attempt + 1,
                        _TTS_MAX_RETRIES,
                        _TTS_RETRY_BASE_DELAY * (2 ** attempt),
                    )
                    if attempt < _TTS_MAX_RETRIES - 1:
                        await asyncio.sleep(_TTS_RETRY_BASE_DELAY * (2 ** attempt))
                elif exc.response.status_code == 401:
                    logger.error("ElevenLabs API key invalid or expired")
                    return None
                else:
                    logger.warning("TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                    if attempt < _TTS_MAX_RETRIES - 1:
                        await asyncio.sleep(_TTS_RETRY_BASE_DELAY)
            except Exception as exc:
                logger.warning("TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_BASE_DELAY)

        if self._last_tts_rate_limited:
            logger.error("ElevenLabs TTS exhausted %d retries on 429; skipping audio for this turn", _TTS_MAX_RETRIES)
        else:
            logger.error("ElevenLabs TTS exhausted %d retries; skipping audio for this turn", _TTS_MAX_RETRIES)
        return None

    @staticmethod
    def _pcm_to_frames(pcm_bytes: bytes) -> list[np.ndarray]:
        """Split raw 16-bit PCM bytes into ~100 ms numpy frames."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return [
            audio[i : i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i : i + _CHUNK_SAMPLES]) > 0
        ]


class LocalSTTElevenLabsHandler(LocalSTTInputMixin, ElevenLabsTTSResponseHandler):
    """Moonshine STT input + ElevenLabs TTS voice output."""

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        # Route explicitly past LocalSTTInputMixin's OpenAI-specific override.
        await ElevenLabsTTSResponseHandler._dispatch_completed_transcript(self, transcript)
