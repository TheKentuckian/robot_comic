"""Chatterbox TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls a
llama-server LLM via /v1/chat/completions for text generation and tool dispatch,
then synthesises audio with the Chatterbox TTS server's /tts endpoint (voice cloning mode).

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
"""

import time
import asyncio
import logging

import numpy as np
from fastrtc import AdditionalOutputs

from robot_comic.config import (
    CHATTERBOX_OUTPUT,
    CHATTERBOX_DEFAULT_URL,
    CHATTERBOX_DEFAULT_GAIN,
    CHATTERBOX_DEFAULT_VOICE,
    CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_TEMPERATURE,
    CHATTERBOX_DEFAULT_EXAGGERATION,
    config,
)
from robot_comic.llama_base import _OUTPUT_SAMPLE_RATE, BaseLlamaResponseHandler, split_sentences
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.chatterbox_voice_clone import load_voice_clone_ref
from robot_comic.chatterbox_tag_translator import translate


logger = logging.getLogger(__name__)

_CHATTERBOX_SAMPLE_RATE = 24000
_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5

# Keep the private alias so the test-suite import (from robot_comic.chatterbox_tts import _split_sentences) still works.
_split_sentences = split_sentences


class ChatterboxTTSResponseHandler(BaseLlamaResponseHandler):
    """llama-server LLM + Chatterbox TTS voice output with tool dispatch."""

    _BACKEND_LABEL = "chatterbox"
    _TTS_SYSTEM = "chatterbox"

    def copy(self) -> "ChatterboxTTSResponseHandler":
        """Create a new instance of this handler with the same configuration."""
        return type(self)(
            self.deps,
            self.sim_mode,
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
        return str(params.get("voice") or getattr(config, "CHATTERBOX_VOICE", CHATTERBOX_DEFAULT_VOICE))

    @property
    def _exaggeration(self) -> float:
        params = self._load_profile_params()
        return float(
            params.get("exaggeration", getattr(config, "CHATTERBOX_EXAGGERATION", CHATTERBOX_DEFAULT_EXAGGERATION))
        )

    @property
    def _cfg_weight(self) -> float:
        params = self._load_profile_params()
        return float(params.get("cfg_weight", getattr(config, "CHATTERBOX_CFG_WEIGHT", CHATTERBOX_DEFAULT_CFG_WEIGHT)))

    @property
    def _temperature(self) -> float:
        params = self._load_profile_params()
        return float(
            params.get("temperature", getattr(config, "CHATTERBOX_TEMPERATURE", CHATTERBOX_DEFAULT_TEMPERATURE))
        )

    @property
    def _gain(self) -> float:
        params = self._load_profile_params()
        return float(params.get("gain", getattr(config, "CHATTERBOX_GAIN", CHATTERBOX_DEFAULT_GAIN)))

    async def _prepare_startup_credentials(self) -> None:
        await super()._prepare_startup_credentials()
        # Resolve per-persona voice-clone reference audio once at startup.
        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if profile:
            self._voice_clone_ref_path = load_voice_clone_ref(config.PROFILES_DIRECTORY / profile)
        else:
            self._voice_clone_ref_path = None
        logger.info(
            "ChatterboxTTS handler initialised: llm=%s/v1/chat/completions tts=%s voice=%s exag=%.2f cfg=%.2f temp=%.2f",
            self._llama_cpp_url,
            self._chatterbox_url,
            self._chatterbox_voice,
            self._exaggeration,
            self._cfg_weight,
            self._temperature,
        )
        await self._warmup_tts()

    async def _warmup_tts(self) -> None:
        """Send a silent warmup request to force the voice model into memory.

        Chatterbox loads the voice model lazily on the first real TTS call,
        which causes a 20-40 s pause on CPU. Doing it here at startup makes
        the cold-start visible in logs and keeps the first user turn snappy.
        """
        logger.info("chatterbox: warming up voice model …")
        _t0 = time.perf_counter()
        try:
            await self._call_chatterbox_tts(".")
        except Exception as exc:
            logger.warning("chatterbox: warmup failed (continuing): %s", exc)
            return
        _dt = time.perf_counter() - _t0
        logger.info("chatterbox: warmup complete in %.1f s", _dt)

    # ------------------------------------------------------------------ #
    # Voice management                                                     #
    # ------------------------------------------------------------------ #

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
        """Return the Chatterbox voice reference file currently in use."""
        return self._chatterbox_voice

    async def change_voice(self, voice: str) -> str:
        """Switch to a different Chatterbox voice reference file."""
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # TTS synthesis                                                        #
    # ------------------------------------------------------------------ #

    async def _synthesize_and_enqueue(self, response_text: str, tts_start: float | None = None) -> None:
        """Translate response_text to TTS segments and enqueue PCM frames."""
        if not response_text:
            return
        from robot_comic import telemetry

        persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default"
        segments = translate(response_text, persona=persona, use_turbo=False)
        any_audio = False
        _first_chunk = True
        for seg in segments:
            if seg.silence_ms:
                for frame in self._pcm_to_frames(self._silence_pcm(seg.silence_ms)):
                    await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
                any_audio = True
            else:
                text = f"{seg.turbo_insert} {seg.text}" if seg.turbo_insert else seg.text
                for sentence in _split_sentences(text):
                    pcm = await self._call_chatterbox_tts(
                        sentence, exaggeration=seg.exaggeration, cfg_weight=seg.cfg_weight
                    )
                    if pcm:
                        if _first_chunk and tts_start is not None:
                            telemetry.record_tts_first_audio(
                                time.perf_counter() - tts_start, {"gen_ai.system": "chatterbox"}
                            )
                            _first_chunk = False
                        for frame in self._pcm_to_frames(pcm):
                            from robot_comic.startup_timer import log_once

                            log_once("first TTS audio frame", logger)
                            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
                        any_audio = True
        if not any_audio:
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": "[TTS error]"}))

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
        # Per-persona voice-clone ref overrides the generic predefined voice when present.
        clone_ref = getattr(self, "_voice_clone_ref_path", None)
        payload: dict[str, object] = {
            "text": text,
            "voice_mode": "clone",
            "output_format": "wav",
            "split_text": False,
            "exaggeration": exaggeration if exaggeration is not None else self._exaggeration,
            "cfg_weight": cfg_weight if cfg_weight is not None else self._cfg_weight,
            "temperature": self._temperature,
        }
        if clone_ref is not None:
            payload["audio_prompt_path"] = str(clone_ref)
        else:
            payload["reference_audio_filename"] = ref_file

        _call_start = time.perf_counter()
        for attempt in range(_TTS_MAX_RETRIES):
            try:
                r = await self._http.post(f"{self._chatterbox_url}/tts", json=payload)
                r.raise_for_status()
                result = self._wav_to_pcm(r.content, gain=self._gain)
                _elapsed = time.perf_counter() - _call_start
                _slow_thresh = float(getattr(config, "REACHY_MINI_TTS_SLOW_WARN_S", 10.0))
                if _elapsed > _slow_thresh:
                    logger.warning(
                        "chatterbox: TTS call took %.1f s (threshold %.0f s) — voice model may have been evicted",
                        _elapsed,
                        _slow_thresh,
                    )
                return result
            except Exception as exc:
                logger.warning(
                    "TTS attempt %d/%d failed: %s: %s", attempt + 1, _TTS_MAX_RETRIES, type(exc).__name__, exc
                )
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_DELAY)
        return None

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes, gain: float = 1.0) -> bytes:
        """Strip WAV header, resample to 24 kHz mono int16 PCM, and apply gain."""
        import io
        import wave

        from scipy.signal import resample

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
        if gain != 1.0:
            audio = audio * gain
        return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


class LocalSTTChatterboxHandler(LocalSTTInputMixin, ChatterboxTTSResponseHandler):
    """Moonshine STT input + Chatterbox TTS voice output."""

    BACKEND_PROVIDER = CHATTERBOX_OUTPUT

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        # Route explicitly past LocalSTTInputMixin's OpenAI-specific override.
        await ChatterboxTTSResponseHandler._dispatch_completed_transcript(self, transcript)
