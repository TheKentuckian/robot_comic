"""llama-server LLM + Gemini 3.1 Flash TTS response handler.

Combines:
- LLM: llama-server /v1/chat/completions (local Qwen3, via ChatterboxTTSResponseHandler)
- TTS: gemini-3.1-flash-tts-preview (cloud, via Gemini client)

Swap-in replacement for LocalSTTGeminiTTSHandler (Gemini LLM + Gemini TTS) or
LocalSTTChatterboxHandler (llama-server LLM + local TTS). Select via:
    LOCAL_STT_RESPONSE_BACKEND=llama_gemini_tts
"""

import asyncio
import base64
import logging
import time
from typing import Any, Optional

import numpy as np
from fastrtc import AdditionalOutputs
from google import genai
from google.genai import types

from robot_comic import telemetry
from robot_comic.chatterbox_tag_translator import translate
from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler, _OUTPUT_SAMPLE_RATE
from robot_comic.config import (
    GEMINI_TTS_AVAILABLE_VOICES,
    GEMINI_TTS_DEFAULT_VOICE,
    LLAMA_GEMINI_TTS_OUTPUT,
    config,
)
from robot_comic.gemini_tts import GEMINI_TTS_MODEL, _TTS_EXCLUSIVE_VOICES
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5

_TTS_SYSTEM_INSTRUCTION = (
    "Deliver this text at a fast, clipped Brooklyn pace — "
    "rapid-fire on the insults, short crisp pauses only where marked. "
    "Never drawl or over-enunciate. Keep the energy sharp."
)


class LlamaGeminiTTSResponseHandler(ChatterboxTTSResponseHandler):
    """llama-server LLM + Gemini 3.1 Flash TTS voice output with tool dispatch."""

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(deps, sim_mode, instance_path, startup_voice)
        self._client: genai.Client | None = None
        # Only restore voices valid for Gemini TTS (not Gemini Live voices)
        self._voice_override = startup_voice if startup_voice in _TTS_EXCLUSIVE_VOICES else None

    def copy(self) -> "LlamaGeminiTTSResponseHandler":
        return LlamaGeminiTTSResponseHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        await super()._prepare_startup_credentials()
        api_key = config.GEMINI_API_KEY or "DUMMY"
        self._client = genai.Client(api_key=api_key)
        logger.info(
            "LlamaGeminiTTS handler initialised: llm=%s/v1/chat/completions tts=%s voice=%s",
            self._llama_cpp_url,
            GEMINI_TTS_MODEL,
            self.get_current_voice(),
        )

    # ------------------------------------------------------------------ #
    # Voice management (Gemini TTS voices, not Chatterbox)                #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        return list(GEMINI_TTS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        voice = self._voice_override or GEMINI_TTS_DEFAULT_VOICE
        if voice not in GEMINI_TTS_AVAILABLE_VOICES:
            return GEMINI_TTS_DEFAULT_VOICE
        return voice

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # TTS synthesis                                                        #
    # ------------------------------------------------------------------ #

    async def _synthesize_and_enqueue(self, response_text: str, tts_start: float | None = None) -> None:
        if not response_text:
            return
        persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default"
        segments = translate(response_text, persona=persona, use_turbo=False)
        clean_text = " ".join(seg.text for seg in segments if seg.text).strip()

        if not clean_text:
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[TTS error]"})
            )
            return

        pcm_bytes = await self._call_gemini_tts(clean_text)
        if pcm_bytes is None:
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[TTS error — Gemini TTS failed]"})
            )
            return

        if tts_start is not None:
            telemetry.record_tts_first_audio(
                time.perf_counter() - tts_start, {"gen_ai.system": "gemini_tts"}
            )
        for frame in self._pcm_to_frames(pcm_bytes):
            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))

    async def _call_gemini_tts(self, text: str) -> bytes | None:
        assert self._client is not None, "Gemini client not initialised"
        tts_config = types.GenerateContentConfig(
            system_instruction=_TTS_SYSTEM_INSTRUCTION,
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
                data = response.candidates[0].content.parts[0].inline_data.data  # type: ignore[index,union-attr]
                return base64.b64decode(data) if isinstance(data, str) else bytes(data)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("Gemini TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_DELAY)
        return None


class LocalSTTLlamaGeminiTTSHandler(LocalSTTInputMixin, LlamaGeminiTTSResponseHandler):  # type: ignore[misc]
    """Moonshine STT input + llama-server LLM + Gemini 3.1 Flash TTS voice output."""

    BACKEND_PROVIDER = LLAMA_GEMINI_TTS_OUTPUT

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        await LlamaGeminiTTSResponseHandler._dispatch_completed_transcript(self, transcript)
