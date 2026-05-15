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
  (moonshine,            elevenlabs, llama)      → LocalSTTLlamaElevenLabsHandler
  (moonshine,            elevenlabs, gemini)     → LocalSTTGeminiElevenLabsHandler
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
from typing import TYPE_CHECKING, Any, Optional, cast

from robot_comic.config import (
    AUDIO_INPUT_HF,
    AUDIO_OUTPUT_HF,
    LLM_BACKEND_ENV,
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    FACTORY_PATH_LEGACY,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_BACKEND_ENV,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    FACTORY_PATH_COMPOSABLE,
    AUDIO_OUTPUT_BACKEND_ENV,
    AUDIO_OUTPUT_GEMINI_LIVE,
    PIPELINE_MODE_COMPOSABLE,
    PIPELINE_MODE_GEMINI_LIVE,
    PIPELINE_MODE_HF_REALTIME,
    AUDIO_INPUT_OPENAI_REALTIME,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    PIPELINE_MODE_OPENAI_REALTIME,
    config,
    derive_pipeline_mode,
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
        pipeline_mode: Optional[str] = None,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> Any:
        """Instantiate and return the correct handler for the given backend pair.

        Args:
            input_backend:  Resolved ``AUDIO_INPUT_BACKEND`` value (e.g. ``"moonshine"``).
            output_backend: Resolved ``AUDIO_OUTPUT_BACKEND`` value (e.g. ``"chatterbox"``).
            deps:           Populated ``ToolDependencies`` instance.
            pipeline_mode:  Optional explicit ``PIPELINE_MODE`` value. When omitted
                the factory derives it from the (input, output) pair so legacy
                call sites keep working unchanged.
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
        handler_kwargs: dict[str, Any] = {
            "deps": deps,
            "sim_mode": sim_mode,
            "instance_path": instance_path,
            "startup_voice": startup_voice,
        }

        # Resolve pipeline_mode: explicit arg > derived from (input, output).
        # When callers (main.py) want config.PIPELINE_MODE to drive selection
        # they pass it explicitly; legacy callers (tests, internal) that just
        # pass an (input, output) pair get the original behaviour via
        # derivation.
        if pipeline_mode is None:
            pipeline_mode = derive_pipeline_mode(input_backend, output_backend)

        # ------------------------------------------------------------------
        # Bundled speech-to-speech sessions (one websocket fuses STT+LLM+TTS).
        # These ignore the input/output dial values; PIPELINE_MODE is the
        # source of truth.
        # ------------------------------------------------------------------

        if pipeline_mode == PIPELINE_MODE_HF_REALTIME:
            from robot_comic.huggingface_realtime import HuggingFaceRealtimeHandler

            logger.info("HandlerFactory: selecting HuggingFaceRealtimeHandler (PIPELINE_MODE=hf_realtime)")
            return HuggingFaceRealtimeHandler(**handler_kwargs)

        if pipeline_mode == PIPELINE_MODE_OPENAI_REALTIME:
            from robot_comic.openai_realtime import OpenaiRealtimeHandler

            logger.info("HandlerFactory: selecting OpenaiRealtimeHandler (PIPELINE_MODE=openai_realtime)")
            return OpenaiRealtimeHandler(**handler_kwargs)

        if pipeline_mode == PIPELINE_MODE_GEMINI_LIVE:
            from robot_comic.gemini_live import GeminiLiveHandler

            logger.info("HandlerFactory: selecting GeminiLiveHandler (PIPELINE_MODE=gemini_live)")
            return GeminiLiveHandler(**handler_kwargs)

        # ------------------------------------------------------------------
        # Composable mode — STT/LLM/TTS dials decide the pipeline.
        # ------------------------------------------------------------------
        assert pipeline_mode == PIPELINE_MODE_COMPOSABLE, f"Unhandled pipeline_mode={pipeline_mode!r}"

        # ------------------------------------------------------------------
        # Local STT (Moonshine) + TTS output pairs — Gemini text-LLM variants
        # ------------------------------------------------------------------

        if input_backend == AUDIO_INPUT_MOONSHINE:
            _llm_backend = getattr(config, "LLM_BACKEND", LLM_BACKEND_LLAMA)

            # LLM_BACKEND=llama variants. The orphan-handler bug was here:
            # before this branch existed the factory ignored LLM_BACKEND=llama
            # for the elevenlabs output and fell through to the
            # Gemini-hardcoded LocalSTTElevenLabsHandler. The llama-specific
            # handlers actually call llama-server for the LLM phase.
            if _llm_backend == LLM_BACKEND_LLAMA:
                if output_backend == AUDIO_OUTPUT_ELEVENLABS:
                    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
                        logger.info(
                            "HandlerFactory: selecting ComposableConversationHandler "
                            "(%s → %s, llm=%s, factory_path=composable)",
                            input_backend,
                            output_backend,
                            LLM_BACKEND_LLAMA,
                        )
                        return _build_composable_llama_elevenlabs(**handler_kwargs)
                    from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler

                    logger.info(
                        "HandlerFactory: selecting LocalSTTLlamaElevenLabsHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_LLAMA,
                    )
                    return LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
                if output_backend == AUDIO_OUTPUT_CHATTERBOX:
                    # Phase 4c.1 (#337): chatterbox+llama is routed through
                    # ComposableConversationHandler when FACTORY_PATH=composable.
                    # Default FACTORY_PATH=legacy falls through to the existing
                    # LocalSTTChatterboxHandler selection in the moonshine block
                    # below.
                    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
                        logger.info(
                            "HandlerFactory: selecting ComposableConversationHandler "
                            "(%s → %s, llm=%s, factory_path=composable)",
                            input_backend,
                            output_backend,
                            LLM_BACKEND_LLAMA,
                        )
                        return _build_composable_llama_chatterbox(**handler_kwargs)
                    # else: fall through to LocalSTTChatterboxHandler below.
                # Other llama+TTS combos (chatterbox legacy path below;
                # gemini_tts has LocalSTTLlamaGeminiTTSHandler) fall through to
                # the existing composable selection below, which picks the
                # correct llama-aware handler for those outputs.

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
                from robot_comic.elevenlabs_tts import LocalSTTGeminiElevenLabsHandler

                logger.info(
                    "HandlerFactory: selecting LocalSTTGeminiElevenLabsHandler (%s → %s)",
                    input_backend,
                    output_backend,
                )
                return LocalSTTGeminiElevenLabsHandler(**handler_kwargs)

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


def _build_composable_llama_elevenlabs(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, llama, elevenlabs) pipeline.

    Builds a legacy ``LocalSTTLlamaElevenLabsHandler`` (the adapters delegate
    into it), wraps it with the three Phase 3 adapters, composes them into a
    ``ComposablePipeline`` seeded with the current session instructions, and
    returns a ``ComposableConversationHandler`` whose ``build`` closure
    re-runs the same construction. FastRTC's ``copy()`` per-peer cloning
    invokes the closure for fresh state on each new peer.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = LlamaLLMAdapter(legacy)
        # LlamaElevenLabsTTSResponseHandler has the same _stream_tts_to_queue +
        # _prepare_startup_credentials surface as ElevenLabsTTSResponseHandler
        # but doesn't share the inheritance chain; the adapter is duck-typed
        # at runtime. Phase 4c will broaden the adapter's annotation.
        tts = ElevenLabsTTSAdapter(cast(Any, legacy))
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=legacy,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()


def _build_composable_llama_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, chatterbox, llama) pipeline.

    Builds a legacy ``LocalSTTChatterboxHandler`` (the adapters delegate into
    it), wraps it with the three Phase 3 adapters, composes them into a
    ``ComposablePipeline`` seeded with the current session instructions, and
    returns a ``ComposableConversationHandler`` whose ``build`` closure
    re-runs the same construction. FastRTC's ``copy()`` per-peer cloning
    invokes the closure for fresh state on each new peer.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ChatterboxTTSAdapter,
    )
    from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = LocalSTTChatterboxHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = LlamaLLMAdapter(legacy)
        tts = ChatterboxTTSAdapter(legacy)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=legacy,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()
