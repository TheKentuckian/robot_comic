"""HandlerFactory: select a concrete conversation handler from the resolved audio backend pair.

This module routes the (input, output, LLM) triple to either a bundled-realtime
handler (HuggingFace / OpenAI Realtime / Gemini Live) or a composable wrapper
(:class:`~robot_comic.composable_conversation_handler.ComposableConversationHandler`)
that hosts the three Phase 3 adapters (STT, LLM, TTS) over the surviving
``*ResponseHandler`` bases. Sub-phase 4e of #337 retired the
``FACTORY_PATH`` dial — composable is now the only path for every triple
except the bundled-realtime fast paths and the LocalSTT+realtime-output
hybrids.

Supported (input, output) → handler-class matrix
--------------------------------------------------
  (hf_input,              hf_output)              → HuggingFaceRealtimeHandler
  (openai_realtime_input, openai_realtime_output) → OpenaiRealtimeHandler
  (gemini_live_input,     gemini_live_output)     → GeminiLiveHandler
  (moonshine,             chatterbox, llama)      → ComposableConversationHandler(ChatterboxTTSResponseHandler)
  (moonshine,             chatterbox, gemini)     → ComposableConversationHandler(GeminiTextChatterboxResponseHandler)
  (moonshine,             elevenlabs, llama)      → ComposableConversationHandler(LlamaElevenLabsTTSResponseHandler)
  (moonshine,             elevenlabs, gemini)     → ComposableConversationHandler(GeminiTextElevenLabsResponseHandler)
  (moonshine,             gemini_tts)             → ComposableConversationHandler(GeminiTTSResponseHandler)
  (moonshine,             openai_realtime_output) → LocalSTTOpenAIRealtimeHandler
  (moonshine,             hf_output)              → LocalSTTHuggingFaceRealtimeHandler

Unsupported combinations raise ``NotImplementedError`` with a message that names
the requested pair and points to docs/audio-backends.md.

Out-of-scope
------------
Arbitrary cross-combinations beyond the supported set would require a proper
Mixin-based handler decomposition.  This factory is strictly a routing layer
over the existing handler classes; cross-combo work is intentionally not
attempted here.
"""

from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from robot_comic.config import (
    AUDIO_INPUT_HF,
    AUDIO_OUTPUT_HF,
    LLM_BACKEND_ENV,
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_BACKEND_ENV,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
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
from robot_comic.backends import ToolCall
from robot_comic.gemini_tts import GeminiTTSResponseHandler
from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
from robot_comic.tools.core_tools import (
    dispatch_tool_call,
    dispatch_tool_call_with_manager,
)
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.gemini_text_handlers import (
    GeminiTextChatterboxResponseHandler,
    GeminiTextElevenLabsResponseHandler,
)
from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler
from robot_comic.tools.tool_constants import SystemTool


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


# ---------------------------------------------------------------------------
# Factory-private mixin host classes.
#
# Each one combines :class:`LocalSTTInputMixin` (the Moonshine STT listener
# the ``MoonshineSTTAdapter`` monkey-patches into) with a surviving
# ``*ResponseHandler`` base (the LLM + TTS implementation the LLM/TTS
# adapters delegate into).
#
# These replace the deleted ``LocalSTT*Handler`` subclasses retired in
# Phase 4e of #337. They are not exported — the composable factory builders
# are their only call site.
#
# Each host re-asserts the ``_dispatch_completed_transcript`` MRO shim
# (``await ResponseHandler._dispatch_completed_transcript(self, transcript)``)
# so the mixin's OpenAI-realtime default (which expects
# ``self.connection``) is not picked by MRO lookup. The composable
# pipeline normally monkey-patches this method via
# :meth:`MoonshineSTTAdapter.start`, but startup-trigger paths and
# direct-dispatch tests reach the method before the patch lands; the
# shim keeps those paths working without surprise.
# ---------------------------------------------------------------------------


class _LocalSTTGeminiChatterboxHost(LocalSTTInputMixin, GeminiTextChatterboxResponseHandler):
    """Composable host: Moonshine STT input + Gemini text LLM + Chatterbox TTS."""

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        await GeminiTextChatterboxResponseHandler._dispatch_completed_transcript(self, transcript)


class _LocalSTTGeminiElevenLabsHost(LocalSTTInputMixin, GeminiTextElevenLabsResponseHandler):
    """Composable host: Moonshine STT input + Gemini text LLM + ElevenLabs TTS."""

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        await GeminiTextElevenLabsResponseHandler._dispatch_completed_transcript(self, transcript)


class _LocalSTTGeminiTTSHost(LocalSTTInputMixin, GeminiTTSResponseHandler):
    """Composable host: Moonshine STT input + bundled Gemini LLM + Gemini TTS."""

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        await GeminiTTSResponseHandler._dispatch_completed_transcript(self, transcript)


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

        if input_backend == AUDIO_INPUT_MOONSHINE:
            _llm_backend = getattr(config, "LLM_BACKEND", LLM_BACKEND_LLAMA)

            # LLM_BACKEND=llama composable triples.
            if _llm_backend == LLM_BACKEND_LLAMA:
                if output_backend == AUDIO_OUTPUT_ELEVENLABS:
                    logger.info(
                        "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_LLAMA,
                    )
                    return _build_composable_llama_elevenlabs(**handler_kwargs)

                if output_backend == AUDIO_OUTPUT_CHATTERBOX:
                    logger.info(
                        "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_LLAMA,
                    )
                    return _build_composable_llama_chatterbox(**handler_kwargs)

                # Llama + gemini_tts is not a supported composable triple
                # today (the GeminiTTSResponseHandler's bundled LLM+TTS
                # design owns its own LLM call and ignores LLM_BACKEND).
                # Fall through to the gemini_tts arm below, which routes
                # via the bundled GeminiBundledLLMAdapter regardless of
                # LLM_BACKEND. Llama + openai_realtime_output and
                # llama + hf_output also fall through to their hybrid
                # selectors below.

            # LLM_BACKEND=gemini composable triples for Chatterbox / ElevenLabs.
            if _llm_backend == LLM_BACKEND_GEMINI:
                if output_backend == AUDIO_OUTPUT_CHATTERBOX:
                    logger.info(
                        "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_GEMINI,
                    )
                    return _build_composable_gemini_chatterbox(**handler_kwargs)

                if output_backend == AUDIO_OUTPUT_ELEVENLABS:
                    logger.info(
                        "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=%s)",
                        input_backend,
                        output_backend,
                        LLM_BACKEND_GEMINI,
                    )
                    return _build_composable_gemini_elevenlabs(**handler_kwargs)

                # Gemini TTS uses its bundled Gemini LLM natively — fall
                # through to the gemini_tts arm below regardless of
                # LLM_BACKEND.
                if output_backend != AUDIO_OUTPUT_GEMINI_TTS:
                    raise NotImplementedError(
                        f"{LLM_BACKEND_ENV}={LLM_BACKEND_GEMINI!r} is not yet implemented "
                        f"for the output backend {AUDIO_OUTPUT_BACKEND_ENV}={output_backend!r}.\n"
                        f"Supported Gemini-text output backends: "
                        f"{AUDIO_OUTPUT_CHATTERBOX!r}, {AUDIO_OUTPUT_ELEVENLABS!r}.\n"
                        f"Set {LLM_BACKEND_ENV}=llama to use the existing llama-server path."
                    )

            # ------------------------------------------------------------------
            # gemini_tts — bundled Gemini LLM + TTS via GeminiTTSResponseHandler.
            # ------------------------------------------------------------------
            if output_backend == AUDIO_OUTPUT_GEMINI_TTS:
                logger.info(
                    "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=gemini-bundled)",
                    input_backend,
                    output_backend,
                )
                return _build_composable_gemini_tts(**handler_kwargs)

            # ------------------------------------------------------------------
            # Llama-fallback chatterbox (LLM_BACKEND neither llama nor gemini).
            # Today's _normalize_llm_backend rejects unknown values so this is
            # the lone "_llm_backend was something exotic" arm — route through
            # the llama-chatterbox builder so behaviour matches the documented
            # default.
            # ------------------------------------------------------------------
            if output_backend == AUDIO_OUTPUT_CHATTERBOX:
                logger.info(
                    "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=llama-fallback)",
                    input_backend,
                    output_backend,
                )
                return _build_composable_llama_chatterbox(**handler_kwargs)

            # ------------------------------------------------------------------
            # Moonshine + ElevenLabs with no llama/gemini selector: route via
            # the gemini-elevenlabs builder. Mirrors the legacy fallback
            # (Gemini was hardcoded in
            # ElevenLabsTTSResponseHandler._prepare_startup_credentials).
            # ------------------------------------------------------------------
            if output_backend == AUDIO_OUTPUT_ELEVENLABS:
                logger.info(
                    "HandlerFactory: selecting ComposableConversationHandler (%s → %s, llm=gemini-fallback)",
                    input_backend,
                    output_backend,
                )
                return _build_composable_gemini_elevenlabs(**handler_kwargs)

            # ------------------------------------------------------------------
            # Moonshine + realtime output hybrids (LocalSTT*RealtimeHandler).
            # The LLM+TTS half lives inside the bundled websocket session, so
            # they don't decompose into the STT/LLM/TTS Protocol triple. Per
            # the operator's Option B decision (Phase 4c-tris Skipped), these
            # hybrids stay legacy forever.
            # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tool-dispatcher shim — Phase 5b of the pipeline refactor.
#
# ``ComposablePipeline.tool_dispatcher`` is the orchestrator's callback for
# turning a model-emitted ``ToolCall`` into the ``str`` result that becomes
# the ``content`` of the next ``role=tool`` history entry. Pre-5b every
# composable factory builder constructed the pipeline with no dispatcher
# argument, so the orchestrator's tool-call branch (composable_pipeline.py:
# 224-231) hit ``self.tool_dispatcher is None``, logged a warning, and
# broke the loop without speaking on any tool-triggered turn — the §2.2
# latent bug from ``docs/superpowers/specs/2026-05-16-phase-5-exploration.md``.
#
# The shim mirrors the routing logic of
# :meth:`robot_comic.tools.background_tool_manager.ToolCallRoutine.__call__`:
# system tools (``task_status`` / ``task_cancel``) need the
# ``BackgroundToolManager`` injected so they can inspect / cancel other
# running tools; everything else goes through the plain dispatcher.
#
# Background tools (per memo §5.2 deferral): legacy ``_start_tool_calls``
# fires a ``BackgroundTool`` and returns immediately, with the result
# arriving via :class:`BackgroundToolManager`'s notification queue. The
# composable orchestrator's synchronous ``await tool_dispatcher(call)``
# model doesn't accommodate that today; we keep the cheapest path
# (synchronous dispatch via ``dispatch_tool_call``) and let any long-running
# tool block this call. This matches the legacy ``_await_tool_results``
# 30 s timeout behaviour for the happy path; the background-tool refactor
# is deferred to a later sub-phase.
#
# The :func:`tool.execute` span emit closes the Rec 1 gap from
# ``docs/superpowers/specs/2026-05-16-instrumentation-audit.md`` — same
# shape as ``background_tool_manager._run_tool``'s span, with the
# ``outcome`` attribute and ``tool.name`` / ``tool.id`` from the
# orchestrator-supplied ``ToolCall``.
# ---------------------------------------------------------------------------


_SYSTEM_TOOL_NAMES: frozenset[str] = frozenset(t.value for t in SystemTool)


def _make_tool_dispatcher(host: Any) -> Any:
    """Build a synchronous tool-dispatcher closure bound to a composable host.

    ``host`` is the factory-private mixin instance (e.g.
    :class:`_LocalSTTLlamaElevenLabsHost`) that owns ``deps`` and a
    :class:`BackgroundToolManager`. The returned coroutine matches
    :data:`robot_comic.composable_pipeline.ToolDispatcher`'s signature.

    The closure emits a ``tool.execute`` span (Phase 5b telemetry, Rec 1
    from the instrumentation audit) and returns the dispatched tool's
    result as a JSON-encoded string — what the orchestrator appends to
    ``conversation_history`` as the ``content`` of a ``role=tool`` entry.
    """
    # Local imports keep ``handler_factory``'s top-level import set lean and
    # avoid a hard dependency on telemetry being initialised at module-load
    # time. The tracer is a no-op when OTel is not initialised.
    from robot_comic import telemetry

    async def _dispatch(call: ToolCall) -> str:
        args = call.args if isinstance(call.args, dict) else {}
        try:
            args_json = json.dumps(args)
        except (TypeError, ValueError):
            # Defensive: a non-serialisable args payload from a model would
            # break the tool layer downstream; surface a clean error string
            # rather than letting the JSON exception escape and break the
            # turn loop.
            logger.warning("ToolCall %s args not JSON-serialisable: %r", call.name, call.args)
            args_json = "{}"

        tracer = telemetry.get_tracer()
        with tracer.start_as_current_span(
            "tool.execute",
            attributes={"tool.name": call.name, "tool.id": call.id},
        ) as span:
            try:
                if call.name in _SYSTEM_TOOL_NAMES:
                    # Mirrors ToolCallRoutine.__call__: system tools need the
                    # BackgroundToolManager so they can inspect / cancel
                    # peers. Reuse the host's manager instance (constructed
                    # in BaseLlamaResponseHandler.__init__:95 and on every
                    # other response-handler base).
                    result = await dispatch_tool_call_with_manager(
                        tool_name=call.name,
                        args_json=args_json,
                        deps=host.deps,
                        tool_manager=host.tool_manager,
                    )
                else:
                    result = await dispatch_tool_call(
                        tool_name=call.name,
                        args_json=args_json,
                        deps=host.deps,
                    )
            except Exception:
                span.set_attribute("outcome", "error")
                raise
            outcome = "error" if isinstance(result, dict) and "error" in result else "success"
            span.set_attribute("outcome", outcome)

        # The orchestrator stores this verbatim as the ``content`` of a
        # ``role=tool`` message; JSON-encode so the LLM sees structured
        # data on the next round-trip (legacy parity:
        # ``llama_base.py:617`` ``json.dumps(result)``).
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            # Last-ditch fallback: surface a string repr if the result is
            # somehow not serialisable. Should never fire — every tool in
            # the registry returns plain dicts.
            logger.warning("Tool %s returned non-JSON-serialisable result", call.name)
            return repr(result)

    return _dispatch


def _maybe_build_welcome_gate() -> Any:
    """Return a :class:`WelcomeGate` for the current profile, or ``None``.

    Mirrors the pre-5e.2 :meth:`LocalSTTInputMixin._build_welcome_gate`
    helper. Returns ``None`` when ``REACHY_MINI_WELCOME_GATE_ENABLED`` is
    false or no profile directory can be resolved.

    Shared across 5e.* triple builders so each migrated triple gets the
    same gating behaviour. The legacy host mirror in
    :class:`LocalSTTInputMixin._build_welcome_gate` survives for the
    un-migrated triples (5e.3-5e.6 retire them one by one).
    """
    from pathlib import Path

    if not getattr(config, "WELCOME_GATE_ENABLED", False):
        return None
    profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    profiles_dir = getattr(config, "PROFILES_DIRECTORY", None)
    if not profile or profiles_dir is None:
        logger.info("welcome gate: enabled but no profile selected — gate inactive")
        return None
    from robot_comic.welcome_gate import make_gate_for_profile

    profile_dir = Path(profiles_dir) / profile
    gate = make_gate_for_profile(profile_dir)
    logger.info("welcome gate: active for profile %r", profile)
    return gate


def _build_composable_llama_elevenlabs(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, llama, elevenlabs) pipeline.

    Phase 5e.2: this triple is the first to migrate off
    :class:`LocalSTTInputMixin`. The handler is now a plain
    :class:`LlamaElevenLabsTTSResponseHandler` (no mixin shell); STT is
    a standalone :class:`MoonshineSTTAdapter` with a
    ``should_drop_frame`` echo-guard closure; the orchestrator-level
    concerns (turn-span, output_queue publishing, set_listening,
    pause-controller, welcome gate, name-validation transcript
    recording) live on :class:`ComposablePipeline` behind the ``deps``
    and ``welcome_gate`` kwargs.

    Subsequent 5e.* triples follow the same pattern; the pipeline-level
    host-concern wiring is shared and needs no further changes.
    """
    import time as _time

    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        # Plain handler — no LocalSTTInputMixin shell. The handler
        # exposes ``_speaking_until`` (set by ``_enqueue_audio_frame``,
        # see ``llama_base.py:233-242``) which the STT echo-guard reads.
        host = LlamaElevenLabsTTSResponseHandler(**handler_kwargs)

        def _should_drop_frame() -> bool:
            # Drop input frames while TTS is still playing — same check
            # the legacy ``LocalSTTInputMixin.receive`` ran inline
            # (``local_stt_realtime.py:796-798``).
            return _time.perf_counter() < getattr(host, "_speaking_until", 0.0)

        stt = MoonshineSTTAdapter(should_drop_frame=_should_drop_frame)
        llm = LlamaLLMAdapter(host)
        # LlamaElevenLabsTTSResponseHandler structurally matches the
        # ElevenLabsTTSAdapter Protocol surface (broadened in 4c.3) even
        # though it doesn't share ElevenLabsTTSResponseHandler's inheritance
        # chain. No cast needed.
        tts = ElevenLabsTTSAdapter(host)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            tool_dispatcher=_make_tool_dispatcher(host),
            system_prompt=get_session_instructions(),
            deps=handler_kwargs["deps"],
            welcome_gate=_maybe_build_welcome_gate(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=host,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()


def _build_composable_llama_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, llama, chatterbox) pipeline.

    Phase 5e.3: this triple migrates off :class:`LocalSTTInputMixin`.
    The handler is now a plain :class:`ChatterboxTTSResponseHandler`
    (no mixin shell); STT is a standalone :class:`MoonshineSTTAdapter`
    with a ``should_drop_frame`` echo-guard closure; the
    orchestrator-level concerns (turn-span, output_queue publishing,
    set_listening, pause-controller, welcome gate, name-validation
    transcript recording) live on :class:`ComposablePipeline` behind
    the ``deps`` and ``welcome_gate`` kwargs.

    Mechanical mirror of :func:`_build_composable_llama_elevenlabs`
    (the Phase 5e.2 sibling); the pipeline-level host-concern wiring
    is shared and needs no further changes.
    """
    import time as _time

    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ChatterboxTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        # Plain handler — no LocalSTTInputMixin shell. The handler
        # exposes ``_speaking_until`` (set by ``_enqueue_audio_frame``,
        # see ``llama_base.py:233-242``) which the STT echo-guard reads.
        host = ChatterboxTTSResponseHandler(**handler_kwargs)

        def _should_drop_frame() -> bool:
            # Drop input frames while TTS is still playing — same check
            # the legacy ``LocalSTTInputMixin.receive`` ran inline
            # (``local_stt_realtime.py:796-798``).
            return _time.perf_counter() < getattr(host, "_speaking_until", 0.0)

        stt = MoonshineSTTAdapter(should_drop_frame=_should_drop_frame)
        llm = LlamaLLMAdapter(host)
        tts = ChatterboxTTSAdapter(host)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            tool_dispatcher=_make_tool_dispatcher(host),
            system_prompt=get_session_instructions(),
            deps=handler_kwargs["deps"],
            welcome_gate=_maybe_build_welcome_gate(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=host,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()


def _build_composable_gemini_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, gemini, chatterbox) pipeline.

    Composes :class:`LocalSTTInputMixin` over
    :class:`GeminiTextChatterboxResponseHandler` via
    :class:`_LocalSTTGeminiChatterboxHost`, wraps it with the three Phase
    3/4 adapters, composes them into a :class:`ComposablePipeline` seeded
    with the current session instructions, and returns a
    :class:`ComposableConversationHandler` whose ``build`` closure re-runs
    the same construction.

    The chatterbox TTS half is shared with the llama variant; the LLM half
    differs only in the adapter (``GeminiLLMAdapter`` wraps
    ``GeminiTextResponseHandler._call_llm`` which Gemini-specifically routes
    through the google-genai SDK while emitting llama-server-shaped
    tool_call dicts upstream).
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ChatterboxTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        host = _LocalSTTGeminiChatterboxHost(**handler_kwargs)
        stt = MoonshineSTTAdapter(host)
        llm = GeminiLLMAdapter(host)
        tts = ChatterboxTTSAdapter(host)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            tool_dispatcher=_make_tool_dispatcher(host),
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=host,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()


def _build_composable_gemini_elevenlabs(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, gemini, elevenlabs) pipeline.

    Composes :class:`LocalSTTInputMixin` over
    :class:`GeminiTextElevenLabsResponseHandler` via
    :class:`_LocalSTTGeminiElevenLabsHost`, wraps it with the three Phase
    3/4 adapters, composes them into a :class:`ComposablePipeline` seeded
    with the current session instructions, and returns a
    :class:`ComposableConversationHandler` whose ``build`` closure re-runs
    the same construction.

    The ElevenLabs TTS half is shared with the llama variant; the LLM half
    is the same ``GeminiLLMAdapter`` from the gemini-chatterbox triple. No
    new adapter is introduced — only the routing. The
    ``_ElevenLabsCompatibleHandler`` Protocol broadening in
    ``elevenlabs_tts_adapter.py`` is what lets the ElevenLabs adapter accept
    the diamond-MRO ``GeminiTextElevenLabsResponseHandler`` without a cast.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        host = _LocalSTTGeminiElevenLabsHost(**handler_kwargs)
        stt = MoonshineSTTAdapter(host)
        llm = GeminiLLMAdapter(host)
        tts = ElevenLabsTTSAdapter(host)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            tool_dispatcher=_make_tool_dispatcher(host),
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=host,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()


def _build_composable_gemini_tts(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, gemini-bundled, gemini_tts) pipeline.

    Composes :class:`LocalSTTInputMixin` over
    :class:`GeminiTTSResponseHandler` via :class:`_LocalSTTGeminiTTSHost`,
    wraps it with the three Phase 3/4 adapters, composes them into a
    :class:`ComposablePipeline` seeded with the current session
    instructions, and returns a :class:`ComposableConversationHandler`
    whose ``build`` closure re-runs the same construction.

    Unlike every other composable triple, the LLM and TTS adapters here both
    wrap the SAME underlying handler instance with a SHARED ``genai.Client``
    (the bundled Gemini-native pattern). The LLM adapter is
    :class:`~robot_comic.adapters.gemini_bundled_llm_adapter.GeminiBundledLLMAdapter`
    (NOT the gemini-text ``GeminiLLMAdapter``) because the handler exposes
    ``_run_llm_with_tools`` rather than ``_call_llm`` — see the Phase 4c.5
    spec for the design rationale.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiTTSAdapter,
        MoonshineSTTAdapter,
        GeminiBundledLLMAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        host = _LocalSTTGeminiTTSHost(**handler_kwargs)
        stt = MoonshineSTTAdapter(host)
        llm = GeminiBundledLLMAdapter(host)
        tts = GeminiTTSAdapter(host)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=host,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()
