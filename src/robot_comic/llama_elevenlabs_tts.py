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
import numpy as np
from fastrtc import AdditionalOutputs

from robot_comic import telemetry
from robot_comic.config import (
    ELEVENLABS_DEFAULT_VOICE,
    ELEVENLABS_AVAILABLE_VOICES,
    LLAMA_ELEVENLABS_TTS_OUTPUT,
    config,
)
from robot_comic.gemini_tts import (
    SHORT_PAUSE_MS,
    SHORT_PAUSE_TAG,
    _silence_pcm,
    extract_delivery_tags,
)
from robot_comic.llama_base import _CHUNK_SAMPLES, _OUTPUT_SAMPLE_RATE, BaseLlamaResponseHandler, split_sentences
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.chatterbox_tag_translator import strip_gemini_tags
from robot_comic.elevenlabs_voices import get_elevenlabs_voices


logger = logging.getLogger(__name__)

_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY = 0.5


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


def apply_voice_settings_deltas(
    base_stability: float,
    base_similarity_boost: float,
    tags: list[str],
) -> dict[str, float]:
    """Map delivery tags to voice_settings adjustments.

    Returns adjusted {stability, similarity_boost} dict, clamped to [0.0, 1.0].
    """
    stability = base_stability
    similarity_boost = base_similarity_boost

    for tag in tags:
        if tag == "fast":
            similarity_boost += 0.2
            stability -= 0.1
        elif tag == "annoyance":
            similarity_boost += 0.3
            stability -= 0.15
        elif tag == "aggression":
            similarity_boost += 0.4
            stability -= 0.2
        elif tag == "slow":
            stability += 0.1
            similarity_boost -= 0.1
        elif tag == "amusement":
            similarity_boost += 0.15
        elif tag == "enthusiasm":
            similarity_boost += 0.2
            stability -= 0.05

    return {
        "stability": max(0.0, min(1.0, stability)),
        "similarity_boost": max(0.0, min(1.0, similarity_boost)),
    }


class LlamaElevenLabsTTSResponseHandler(BaseLlamaResponseHandler):
    """llama-server LLM + ElevenLabs TTS voice output with tool dispatch."""

    _BACKEND_LABEL = "llama_elevenlabs_tts"
    _TTS_SYSTEM = "elevenlabs"
    # ElevenLabs Turbo v2.5 pricing: $0.50 per 1M characters (Creator tier)
    # verify against current ElevenLabs pricing
    ELEVENLABS_COST_PER_1M_CHARS: float = 0.50

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
        self.cumulative_cost: float = 0.0

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
        Otherwise map the named voice via the dynamic voice catalog.
        """
        config_params = load_profile_elevenlabs_config()
        custom_id = config_params.get("voice_id")
        if custom_id:
            return custom_id
        voice_name = self.get_current_voice()
        voice_catalog = get_elevenlabs_voices()
        return voice_catalog.get(voice_name)

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
        # List used as a one-shot first-audio marker shared with _stream_tts_to_queue.
        # When non-empty, the first PCM chunk fires record_tts_first_audio and clears it.
        first_audio_marker: list[float] = [tts_start] if tts_start is not None else []
        for sentence in sentences:
            # Extract delivery tags before stripping so they can guide voice_settings.
            tags = extract_delivery_tags(sentence)
            # Strip Gemini-style delivery tags ([fast], [annoyance], etc.) so they
            # aren't spoken literally. [short pause] becomes a real silence gap.
            spoken = strip_gemini_tags(sentence)
            if not spoken:
                continue
            if SHORT_PAUSE_TAG in tags:
                for frame in self._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS, _OUTPUT_SAMPLE_RATE)):
                    await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
            sentence_had_audio = await self._stream_tts_to_queue(spoken, first_audio_marker, tags)
            if sentence_had_audio:
                any_audio = True

        if not any_audio:
            if self._last_tts_rate_limited:
                msg = "[ElevenLabs TTS rate-limited; try again later]"
            else:
                msg = "[TTS error — ElevenLabs TTS failed]"
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": msg}))

    async def _stream_tts_to_queue(
        self,
        text: str,
        first_audio_marker: list[float] | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Stream ElevenLabs TTS PCM chunks directly into ``output_queue``.

        Uses the ``/stream`` endpoint with ``output_format=pcm_24000`` so first
        audio arrives in ~100-200ms instead of ~500-1000ms with full-body POST.

        ``first_audio_marker``: when non-empty, the first PCM chunk fires
        ``telemetry.record_tts_first_audio`` (perf_counter - marker[0]) and the
        marker is cleared so subsequent sentences in the same turn don't refire.

        ``tags``: delivery tags to adjust voice_settings (e.g., [fast], [annoyance]).

        Returns True if any audio was streamed for this call.
        """
        assert self._http is not None, "HTTP client not initialised"

        api_key = config.ELEVENLABS_API_KEY
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not configured")
            return False

        voice_id = self._resolve_voice_id()
        if not voice_id:
            logger.error("Could not resolve voice ID for %s", self.get_current_voice())
            return False

        # Accumulate cost: ElevenLabs charges per character (text is already tag-stripped).
        char_count = len(text)
        cost = (char_count / 1_000_000) * self.ELEVENLABS_COST_PER_1M_CHARS
        self.cumulative_cost += cost
        if cost > 0:
            logger.debug("ElevenLabs TTS cost: $%.4f (%d chars) | Cumulative: $%.4f", cost, char_count, self.cumulative_cost)

        config_params = load_profile_elevenlabs_config()
        base_stability = float(config_params.get("stability", "0.5"))
        base_similarity_boost = float(config_params.get("similarity_boost", "0.75"))

        # Apply per-sentence tag adjustments to voice_settings.
        voice_settings = apply_voice_settings_deltas(
            base_stability,
            base_similarity_boost,
            tags or [],
        )

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_{_OUTPUT_SAMPLE_RATE}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": voice_settings,
        }

        self._last_tts_rate_limited = False
        frame_bytes = _CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample
        for attempt in range(_TTS_MAX_RETRIES):
            try:
                got_audio = False
                leftover = b""
                async with self._http.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        if not got_audio:
                            got_audio = True
                            if first_audio_marker:
                                telemetry.record_tts_first_audio(
                                    time.perf_counter() - first_audio_marker[0],
                                    {"gen_ai.system": "elevenlabs"},
                                )
                                first_audio_marker.clear()
                        leftover += chunk
                        while len(leftover) >= frame_bytes:
                            frame = np.frombuffer(leftover[:frame_bytes], dtype=np.int16)
                            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
                            leftover = leftover[frame_bytes:]
                if leftover:
                    tail = np.frombuffer(leftover[: (len(leftover) // 2) * 2], dtype=np.int16)
                    if len(tail) > 0:
                        await self.output_queue.put((_OUTPUT_SAMPLE_RATE, tail))
                return got_audio
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    self._last_tts_rate_limited = True
                    logger.warning(
                        "ElevenLabs TTS 429 (attempt %d/%d); sleeping %.1fs before retry",
                        attempt + 1,
                        _TTS_MAX_RETRIES,
                        _TTS_RETRY_BASE_DELAY * (2**attempt),
                    )
                    if attempt < _TTS_MAX_RETRIES - 1:
                        await asyncio.sleep(_TTS_RETRY_BASE_DELAY * (2**attempt))
                elif exc.response.status_code == 401:
                    logger.error("ElevenLabs API key invalid or expired")
                    return False
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
        return False


class LocalSTTLlamaElevenLabsHandler(LocalSTTInputMixin, LlamaElevenLabsTTSResponseHandler):  # type: ignore[misc]
    """Moonshine STT input + llama-server LLM + ElevenLabs TTS voice output."""

    BACKEND_PROVIDER = LLAMA_ELEVENLABS_TTS_OUTPUT

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        await LlamaElevenLabsTTSResponseHandler._dispatch_completed_transcript(self, transcript)
