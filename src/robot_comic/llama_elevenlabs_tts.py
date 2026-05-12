"""llama-server LLM + ElevenLabs TTS response handler.

Combines:
- LLM: llama-server /v1/chat/completions (local Qwen3)
- TTS: ElevenLabs API (cloud)

Select via:
    LOCAL_STT_RESPONSE_BACKEND=llama_elevenlabs_tts
"""

import time
import asyncio
import logging
from typing import Optional

import httpx
from fastrtc import AdditionalOutputs

from robot_comic import telemetry
from robot_comic.config import (
    LLAMA_ELEVENLABS_TTS_OUTPUT,
    ELEVENLABS_DEFAULT_VOICE,
    ELEVENLABS_AVAILABLE_VOICES,
    config,
)
from robot_comic.llama_base import _OUTPUT_SAMPLE_RATE, BaseLlamaResponseHandler, split_sentences
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY = 0.5

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


class LlamaElevenLabsTTSResponseHandler(BaseLlamaResponseHandler):
    """llama-server LLM + ElevenLabs TTS voice output with tool dispatch."""

    _BACKEND_LABEL = "llama_elevenlabs_tts"
    _TTS_SYSTEM = "elevenlabs"

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(deps, sim_mode, instance_path, startup_voice)
        self._http: httpx.AsyncClient | None = None
        self._last_tts_rate_limited: bool = False

    def copy(self) -> "LlamaElevenLabsTTSResponseHandler":
        return LlamaElevenLabsTTSResponseHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        await super()._prepare_startup_credentials()
        self._http = httpx.AsyncClient(timeout=30.0)

        llm_model = await self._fetch_llm_model_name()
        stt_model = getattr(config, "LOCAL_STT_MODEL", "unknown")
        logger.info(
            "Pipeline: Moonshine (%s) → llama-server (%s @ %s) → ElevenLabs TTS (voice=%s)",
            stt_model,
            llm_model,
            self._llama_cpp_url,
            self.get_current_voice(),
        )

    async def _fetch_llm_model_name(self) -> str:
        assert self._http is not None
        try:
            r = await self._http.get(f"{self._llama_cpp_url}/v1/models", timeout=3.0)
            r.raise_for_status()
            data = r.json()
            return data["data"][0]["id"]
        except Exception:
            return self._llama_cpp_url

    # ------------------------------------------------------------------ #
    # Voice management (ElevenLabs voices)                                #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        return list(ELEVENLABS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        if self._voice_override:
            return self._voice_override
        config_params = load_profile_elevenlabs_config()
        voice = config_params.get("voice") or ELEVENLABS_DEFAULT_VOICE
        # Custom voice_id (e.g. PVC clone) takes precedence; the "voice" name is
        # informational when voice_id is set.
        if config_params.get("voice_id"):
            return voice
        if voice not in ELEVENLABS_AVAILABLE_VOICES:
            logger.warning(
                "Voice %r is not a valid ElevenLabs voice; falling back to %s", voice, ELEVENLABS_DEFAULT_VOICE
            )
            return ELEVENLABS_DEFAULT_VOICE
        return voice

    def _resolve_voice_id(self) -> str | None:
        """Resolve the ElevenLabs voice ID.

        Profile config `voice_id=<id>` takes precedence (e.g. PVC clones).
        Otherwise map the named voice via the prebuilt voice catalog.
        """
        config_params = load_profile_elevenlabs_config()
        custom_id = config_params.get("voice_id")
        if custom_id:
            return custom_id
        return _ELEVENLABS_VOICE_IDS.get(self.get_current_voice())

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # TTS synthesis                                                        #
    # ------------------------------------------------------------------ #

    async def _synthesize_and_enqueue(self, response_text: str, tts_start: float | None = None) -> None:
        if not response_text:
            return
        sentences = split_sentences(response_text) or [response_text]
        any_audio = False
        first_chunk = True
        for sentence in sentences:
            if not sentence:
                continue
            pcm_bytes = await self._call_elevenlabs_tts(sentence)
            if pcm_bytes is None:
                logger.warning("ElevenLabs TTS returned None for sentence: %r", sentence[:60])
                continue
            if first_chunk and tts_start is not None:
                telemetry.record_tts_first_audio(time.perf_counter() - tts_start, {"gen_ai.system": "elevenlabs"})
                first_chunk = False
            for frame in self._pcm_to_frames(pcm_bytes):
                await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
            any_audio = True

        if not any_audio:
            if self._last_tts_rate_limited:
                msg = "[ElevenLabs TTS rate-limited; try again later]"
            else:
                msg = "[TTS error — ElevenLabs TTS failed]"
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": msg}))

    async def _call_elevenlabs_tts(self, text: str) -> bytes | None:
        assert self._http is not None, "HTTP client not initialised"

        api_key = config.ELEVENLABS_API_KEY
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not configured")
            return None

        voice_id = self._resolve_voice_id()
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
                    logger.warning("ElevenLabs TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                    if attempt < _TTS_MAX_RETRIES - 1:
                        await asyncio.sleep(_TTS_RETRY_BASE_DELAY)
            except Exception as exc:
                logger.warning("ElevenLabs TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_BASE_DELAY)

        if self._last_tts_rate_limited:
            logger.error("ElevenLabs TTS exhausted %d retries on 429; skipping audio for this turn", _TTS_MAX_RETRIES)
        else:
            logger.error("ElevenLabs TTS exhausted %d retries; skipping audio for this turn", _TTS_MAX_RETRIES)
        return None


class LocalSTTLlamaElevenLabsHandler(LocalSTTInputMixin, LlamaElevenLabsTTSResponseHandler):  # type: ignore[misc]
    """Moonshine STT input + llama-server LLM + ElevenLabs TTS voice output."""

    BACKEND_PROVIDER = LLAMA_ELEVENLABS_TTS_OUTPUT

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        await LlamaElevenLabsTTSResponseHandler._dispatch_completed_transcript(self, transcript)
