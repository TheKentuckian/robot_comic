"""HandlerFactory: select a concrete conversation handler from the resolved audio backend pair.

This module is the bridge between the modular audio config scaffold introduced
in PR #215 and the concrete handler classes that were previously selected via a
BACKEND_PROVIDER conditional in main.py.

Supported (input, output) → handler-class matrix
--------------------------------------------------
  (hf_input,             hf_output)             → HuggingFaceRealtimeHandler
  (openai_realtime_input, openai_realtime_output)→ OpenaiRealtimeHandler
  (gemini_live_input,    gemini_live_output)     → GeminiLiveHandler
  (moonshine,            chatterbox)             → LocalSTTChatterboxHandler
  (moonshine,            gemini_tts)             → LocalSTTGeminiTTSHandler
  (moonshine,            elevenlabs)             → LocalSTTElevenLabsHandler
  (moonshine,            openai_realtime_output) → LocalSTTOpenAIRealtimeHandler
  (moonshine,            hf_output)              → LocalSTTHuggingFaceRealtimeHandler

Unsupported combinations raise ``NotImplementedError`` with a message that names
the requested pair and points to docs/audio-backends.md.

Out-of-scope
------------
Arbitrary cross-combinations beyond the supported set would require a proper
Mixin-based handler decomposition.  This factory is strictly a routing layer
over the *existing* handler classes; cross-combo work is intentionally not
attempted here.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Optional

from robot_comic.config import (
    AUDIO_INPUT_HF,
    AUDIO_OUTPUT_HF,
    LLM_BACKEND_ENV,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_BACKEND_ENV,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_BACKEND_ENV,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_INPUT_OPENAI_REALTIME,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    config,
)


if TYPE_CHECKING:
    from robot_comic.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

# Human-readable names for log / error messages.
_INPUT_NAMES: dict[str, str] = {
    AUDIO_INPUT_HF: "Hugging Face realtime",
    AUDIO_INPUT_OPENAI_REALTIME: "OpenAI Realtime",
    AUDIO_INPUT_GEMINI_LIVE: "Gemini Live",
    AUDIO_INPUT_MOONSHINE: "Moonshine (local STT)",
}
_OUTPUT_NAMES: dict[str, str] = {
    AUDIO_OUTPUT_HF: "Hugging Face realtime",
    AUDIO_OUTPUT_OPENAI_REALTIME: "OpenAI Realtime",
    AUDIO_OUTPUT_GEMINI_LIVE: "Gemini Live",
    AUDIO_OUTPUT_CHATTERBOX: "Chatterbox TTS",
    AUDIO_OUTPUT_GEMINI_TTS: "Gemini TTS",
    AUDIO_OUTPUT_ELEVENLABS: "ElevenLabs TTS",
}

_SUPPORTED_MATRIX_DOC = (
    "Supported combinations:\n"
    f"  ({AUDIO_INPUT_HF}, {AUDIO_OUTPUT_HF})\n"
    f"  ({AUDIO_INPUT_OPENAI_REALTIME}, {AUDIO_OUTPUT_OPENAI_REALTIME})\n"
    f"  ({AUDIO_INPUT_GEMINI_LIVE}, {AUDIO_OUTPUT_GEMINI_LIVE})\n"
    f"  ({AUDIO_INPUT_MOONSHINE}, {AUDIO_OUTPUT_CHATTERBOX})\n"
    f"  ({AUDIO_INPUT_MOONSHINE}, {AUDIO_OUTPUT_GEMINI_TTS})\n"
    f"  ({AUDIO_INPUT_MOONSHINE}, {AUDIO_OUTPUT_ELEVENLABS})\n"
    f"  ({AUDIO_INPUT_MOONSHINE}, {AUDIO_OUTPUT_OPENAI_REALTIME})\n"
    f"  ({AUDIO_INPUT_MOONSHINE}, {AUDIO_OUTPUT_HF})\n"
    "See docs/audio-backends.md for details."
)


class HandlerFactory:
    """Factory that maps a resolved (input_backend, output_backend) pair to a handler."""

    @staticmethod
    def build(
        input_backend: str,
        output_backend: str,
        deps: "ToolDependencies",
        *,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> Any:
        """Instantiate and return the correct handler for the given backend pair.

        Args:
            input_backend:  Resolved ``AUDIO_INPUT_BACKEND`` value (e.g. ``"moonshine"``).
            output_backend: Resolved ``AUDIO_OUTPUT_BACKEND`` value (e.g. ``"chatterbox"``).
            deps:           Populated ``ToolDependencies`` instance.
            sim_mode:       Whether the app is running in simulation / dev mode.
            instance_path:  Optional path to the per-instance data directory.
            startup_voice:  Optional voice name loaded from ``startup_settings.json``.

        Returns:
            An instantiated handler object ready to be passed to FastRTC / LocalStream.

        Raises:
            NotImplementedError: If ``(input_backend, output_backend)`` is not a
                supported combination.  The error message lists all supported pairs
                and references ``docs/audio-backends.md``.

        """
        combo = (input_backend, output_backend)

        handler_kwargs: dict[str, Any] = {
            "deps": deps,
            "sim_mode": sim_mode,
            "instance_path": instance_path,
            "startup_voice": startup_voice,
        }

        # ------------------------------------------------------------------
        # Fully-realtime bundled pairs
        # ------------------------------------------------------------------

        if combo == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF):
            from robot_comic.huggingface_realtime import HuggingFaceRealtimeHandler

            logger.info(
                "HandlerFactory: selecting HuggingFaceRealtimeHandler (%s → %s)",
                input_backend,
                output_backend,
            )
            return HuggingFaceRealtimeHandler(**handler_kwargs)

        if combo == (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME):
            from robot_comic.openai_realtime import OpenaiRealtimeHandler

            logger.info(
                "HandlerFactory: selecting OpenaiRealtimeHandler (%s → %s)",
                input_backend,
                output_backend,
            )
            return OpenaiRealtimeHandler(**handler_kwargs)

        if combo == (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE):
            from robot_comic.gemini_live import GeminiLiveHandler

            logger.info(
                "HandlerFactory: selecting GeminiLiveHandler (%s → %s)",
                input_backend,
                output_backend,
            )
            return GeminiLiveHandler(**handler_kwargs)

        # ------------------------------------------------------------------
        # Local STT (Moonshine) + TTS output pairs — Gemini text-LLM variants
        # ------------------------------------------------------------------

        if input_backend == AUDIO_INPUT_MOONSHINE:
            _llm_backend = getattr(config, "LLM_BACKEND", "llama")
            if _llm_backend == LLM_BACKEND_GEMINI:
                if output_backend == AUDIO_OUTPUT_CHATTERBOX:
                    from robot_comic.gemini_text_handlers import GeminiTextChatterboxHandler

                    logger.info(
                        "HandlerFactory: selecting GeminiTextChatterboxHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_GEMINI,
                    )
                    return GeminiTextChatterboxHandler(**handler_kwargs)

                if output_backend == AUDIO_OUTPUT_ELEVENLABS:
                    from robot_comic.gemini_text_handlers import GeminiTextElevenLabsHandler

                    logger.info(
                        "HandlerFactory: selecting GeminiTextElevenLabsHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_GEMINI,
                    )
                    return GeminiTextElevenLabsHandler(**handler_kwargs)

                # Gemini TTS: already uses Gemini for LLM natively — fall through
                # to the llama routing so LocalSTTGeminiTTSHandler is selected.
                if output_backend != AUDIO_OUTPUT_GEMINI_TTS:
                    raise NotImplementedError(
                        f"{LLM_BACKEND_ENV}={LLM_BACKEND_GEMINI!r} is not yet implemented "
                        f"for the output backend {AUDIO_OUTPUT_BACKEND_ENV}={output_backend!r}.\n"
                        f"Supported Gemini-text output backends: "
                        f"{AUDIO_OUTPUT_CHATTERBOX!r}, {AUDIO_OUTPUT_ELEVENLABS!r}.\n"
                        f"Set {LLM_BACKEND_ENV}=llama to use the existing llama-server path."
                    )

        # ------------------------------------------------------------------
        # Local STT (Moonshine) + TTS output pairs
        # ------------------------------------------------------------------

        if input_backend == AUDIO_INPUT_MOONSHINE:
            if output_backend == AUDIO_OUTPUT_CHATTERBOX:
                from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler

                logger.info(
                    "HandlerFactory: selecting LocalSTTChatterboxHandler (%s → %s)",
                    input_backend,
                    output_backend,
                )
                return LocalSTTChatterboxHandler(**handler_kwargs)

            if output_backend == AUDIO_OUTPUT_GEMINI_TTS:
                from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler

                logger.info(
                    "HandlerFactory: selecting LocalSTTGeminiTTSHandler (%s → %s)",
                    input_backend,
                    output_backend,
                )
                return LocalSTTGeminiTTSHandler(**handler_kwargs)

            if output_backend == AUDIO_OUTPUT_ELEVENLABS:
                from robot_comic.elevenlabs_tts import LocalSTTElevenLabsHandler

                logger.info(
                    "HandlerFactory: selecting LocalSTTElevenLabsHandler (%s → %s)",
                    input_backend,
                    output_backend,
                )
                return LocalSTTElevenLabsHandler(**handler_kwargs)

            if output_backend == AUDIO_OUTPUT_OPENAI_REALTIME:
                from robot_comic.local_stt_realtime import LocalSTTOpenAIRealtimeHandler

                logger.info(
                    "HandlerFactory: selecting LocalSTTOpenAIRealtimeHandler (%s → %s)",
                    input_backend,
                    output_backend,
                )
                return LocalSTTOpenAIRealtimeHandler(**handler_kwargs)

            if output_backend == AUDIO_OUTPUT_HF:
                from robot_comic.local_stt_realtime import LocalSTTHuggingFaceRealtimeHandler

                logger.info(
                    "HandlerFactory: selecting LocalSTTHuggingFaceRealtimeHandler (%s → %s)",
                    input_backend,
                    output_backend,
                )
                return LocalSTTHuggingFaceRealtimeHandler(**handler_kwargs)

        # ------------------------------------------------------------------
        # Unsupported combination
        # ------------------------------------------------------------------
        in_label = _INPUT_NAMES.get(input_backend, repr(input_backend))
        out_label = _OUTPUT_NAMES.get(output_backend, repr(output_backend))
        raise NotImplementedError(
            f"No handler implementation exists for the audio backend combination "
            f"{AUDIO_INPUT_BACKEND_ENV}={input_backend!r} ({in_label}) + "
            f"{AUDIO_OUTPUT_BACKEND_ENV}={output_backend!r} ({out_label}).\n"
            f"{_SUPPORTED_MATRIX_DOC}\n"
            "Arbitrary cross-combinations are not supported by the current "
            "handler classes; see docs/audio-backends.md for the supported set."
        )
