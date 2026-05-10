"""Chatterbox TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls an
OpenAI-compatible LLM (Ollama by default) for text generation, then synthesises
audio with the Chatterbox TTS server's /tts endpoint (voice cloning mode).

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
Tool call support is deferred; see GitHub issue #39.
"""

import asyncio
import logging
from typing import Any, Optional

import httpx
import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item
from scipy.signal import resample

from robot_comic.chatterbox_tag_translator import translate
from robot_comic.config import (
    CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_EXAGGERATION,
    CHATTERBOX_DEFAULT_TEMPERATURE,
    CHATTERBOX_DEFAULT_URL,
    CHATTERBOX_DEFAULT_VOICE,
    CHATTERBOX_OUTPUT,
    config,
    set_custom_profile,
)
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.prompts import get_session_instructions
from robot_comic.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400          # 100 ms at 24 kHz
_CHATTERBOX_SAMPLE_RATE = 24000
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0
_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5


class ChatterboxTTSResponseHandler(ConversationHandler):
    """OpenAI-compatible LLM + Chatterbox TTS voice output (text-only, no tools)."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override: str | None = startup_voice
        self._http: httpx.AsyncClient | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        self._llm_context: list[int] | None = None  # Ollama context tokens
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Attributes referenced by LocalSTTInputMixin
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    def _mark_activity(self, label: str) -> None:
        logger.debug("Activity: %s", label)

    def copy(self) -> "ChatterboxTTSResponseHandler":
        return ChatterboxTTSResponseHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    def _load_profile_params(self) -> dict[str, str]:
        """Read profiles/<name>/chatterbox.txt as key=value pairs, if present."""
        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if not profile:
            return {}
        try:
            path = config.PROFILES_DIRECTORY / profile / "chatterbox.txt"
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
            logger.warning("Could not read chatterbox.txt for profile %r: %s", profile, exc)
            return {}

    @property
    def _chatterbox_url(self) -> str:
        return getattr(config, "CHATTERBOX_URL", CHATTERBOX_DEFAULT_URL)

    @property
    def _chatterbox_voice(self) -> str:
        if self._voice_override:
            return self._voice_override
        params = self._load_profile_params()
        return params.get("voice") or getattr(config, "CHATTERBOX_VOICE", CHATTERBOX_DEFAULT_VOICE)

    @property
    def _exaggeration(self) -> float:
        params = self._load_profile_params()
        return float(params.get("exaggeration", getattr(config, "CHATTERBOX_EXAGGERATION", CHATTERBOX_DEFAULT_EXAGGERATION)))

    @property
    def _cfg_weight(self) -> float:
        params = self._load_profile_params()
        return float(params.get("cfg_weight", getattr(config, "CHATTERBOX_CFG_WEIGHT", CHATTERBOX_DEFAULT_CFG_WEIGHT)))

    @property
    def _temperature(self) -> float:
        params = self._load_profile_params()
        return float(params.get("temperature", getattr(config, "CHATTERBOX_TEMPERATURE", CHATTERBOX_DEFAULT_TEMPERATURE)))

    @property
    def _ollama_base_url(self) -> str:
        import urllib.parse
        parsed = urllib.parse.urlparse(self._chatterbox_url)
        return f"{parsed.scheme}://{parsed.hostname}:11434"

    async def _prepare_startup_credentials(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        )
        logger.info(
            "ChatterboxTTS handler initialised: llm=%s/api/generate tts=%s voice=%s exag=%.2f cfg=%.2f temp=%.2f",
            self._ollama_base_url,
            self._chatterbox_url,
            self._chatterbox_voice,
            self._exaggeration,
            self._cfg_weight,
            self._temperature,
        )

    async def start_up(self) -> None:
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        asyncio.create_task(self._send_startup_trigger(), name="chatterbox-startup-trigger")
        await self._stop_event.wait()

    async def _send_startup_trigger(self) -> None:
        await self._dispatch_completed_transcript("[conversation started]")

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive(self, frame: Any) -> None:
        """No-op: audio input is handled by LocalSTTInputMixin.receive()."""

    async def emit(self) -> Any:
        return await wait_for_item(self.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        try:
            set_custom_profile(profile)
            self._conversation_history.clear()
            self._llm_context = None
            return f"Applied personality {profile!r}. Conversation history reset."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        """Return predefined voice filenames from the Chatterbox server."""
        assert self._http is not None
        try:
            r = await self._http.get(f"{self._chatterbox_url}/get_predefined_voices")
            r.raise_for_status()
            return [v["filename"] for v in r.json()]
        except Exception as exc:
            logger.warning("Could not fetch Chatterbox voices: %s", exc)
            return [self._chatterbox_voice]

    def get_current_voice(self) -> str:
        return self._chatterbox_voice

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # Response cycle                                                       #
    # ------------------------------------------------------------------ #

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """LLM → TTS → PCM frames."""
        self._conversation_history.append({"role": "user", "content": transcript})

        try:
            response_text = await self._call_llm()
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return

        self._conversation_history.append({"role": "assistant", "content": response_text})
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": response_text})
        )

        persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default"
        segments = translate(response_text, persona=persona, use_turbo=False)

        pcm_parts: list[bytes] = []
        for seg in segments:
            if seg.silence_ms:
                pcm_parts.append(self._silence_pcm(seg.silence_ms))
            else:
                text = f"{seg.turbo_insert} {seg.text}" if seg.turbo_insert else seg.text
                pcm = await self._call_chatterbox_tts(
                    text, exaggeration=seg.exaggeration, cfg_weight=seg.cfg_weight
                )
                if pcm:
                    pcm_parts.append(pcm)

        if not pcm_parts:
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[TTS error]"})
            )
            return

        for frame in self._pcm_to_frames(b"".join(pcm_parts)):
            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))

    async def _call_llm(self) -> str:
        """Call Ollama /api/generate; maintain conversation via context tokens."""
        assert self._http is not None
        system_prompt = get_session_instructions()
        latest_user_msg = self._conversation_history[-1]["content"]

        payload: dict[str, Any] = {
            "model": getattr(config, "MODEL_NAME", "hermes3:8b-llama3.1-q4_K_M"),
            "system": system_prompt,
            "prompt": latest_user_msg,
            "stream": False,
        }
        if self._llm_context:
            payload["context"] = self._llm_context

        delay = _LLM_RETRY_BASE_DELAY
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                r = await self._http.post(
                    f"{self._ollama_base_url}/api/generate",
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                self._llm_context = data.get("context")
                return data["response"].strip()
            except Exception as exc:
                if attempt == _LLM_MAX_RETRIES - 1:
                    raise
                logger.warning("LLM attempt %d/%d failed: %s: %s; retrying in %.1fs",
                               attempt + 1, _LLM_MAX_RETRIES, type(exc).__name__, exc, delay)
                await asyncio.sleep(delay)
                delay *= 2
        return ""

    async def _call_chatterbox_tts(
        self,
        text: str,
        *,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> bytes | None:
        """POST to Chatterbox /tts in clone mode; return raw WAV bytes."""
        assert self._http is not None
        voice = self._chatterbox_voice
        ref_file = voice if voice.endswith(".wav") else f"{voice}.wav"
        payload = {
            "text": text,
            "voice_mode": "clone",
            "reference_audio_filename": ref_file,
            "output_format": "wav",
            "exaggeration": exaggeration if exaggeration is not None else self._exaggeration,
            "cfg_weight": cfg_weight if cfg_weight is not None else self._cfg_weight,
            "temperature": self._temperature,
        }

        for attempt in range(_TTS_MAX_RETRIES):
            try:
                r = await self._http.post(f"{self._chatterbox_url}/tts", json=payload)
                r.raise_for_status()
                return self._wav_to_pcm(r.content)
            except Exception as exc:
                logger.warning("TTS attempt %d/%d failed: %s: %s", attempt + 1, _TTS_MAX_RETRIES, type(exc).__name__, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_DELAY)
        return None

    @staticmethod
    def _silence_pcm(duration_ms: int) -> bytes:
        n_samples = int(_OUTPUT_SAMPLE_RATE * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.int16).tobytes()

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        """Strip WAV header and resample to 24 kHz mono int16 PCM."""
        import wave, io
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            src_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        if src_rate != _OUTPUT_SAMPLE_RATE:
            target_len = int(len(audio) * _OUTPUT_SAMPLE_RATE / src_rate)
            audio = resample(audio, target_len)
        return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _pcm_to_frames(pcm_bytes: bytes) -> list[np.ndarray]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return [
            audio[i: i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i: i + _CHUNK_SAMPLES]) > 0
        ]


class LocalSTTChatterboxHandler(LocalSTTInputMixin, ChatterboxTTSResponseHandler):
    """Moonshine STT input + Chatterbox TTS voice output."""

    BACKEND_PROVIDER = CHATTERBOX_OUTPUT

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        # Route explicitly past LocalSTTInputMixin's OpenAI-specific override (same
        # pattern as LocalSTTGeminiTTSHandler).
        await ChatterboxTTSResponseHandler._dispatch_completed_transcript(self, transcript)
