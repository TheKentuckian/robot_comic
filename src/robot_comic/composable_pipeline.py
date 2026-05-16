"""Composable conversation pipeline — Phase 2 of the pipeline refactor.

Orchestrates a 3-phase pipeline (STT → LLM → TTS) by composing backend
objects that satisfy the Protocols defined in :mod:`robot_comic.backends`
(Phase 1). This sits alongside the existing ``ConversationHandler`` ABC in
:mod:`robot_comic.conversation_handler`; it does NOT replace it yet.

The existing ABC's concrete subclasses (``BaseLlamaResponseHandler``,
``GeminiTextResponseHandler``, ``ElevenLabsTTSResponseHandler``, etc.)
express their 3-phase pipeline via inheritance — every ``(STT, LLM, TTS)``
triple is a distinct class with the LLM dial baked into the class
hierarchy. After Phase 3, swapping the LLM means swapping one object, not
introducing a new subclass.

## Phase 2 scope (this PR)

Net-new code only. Operators don't yet use ``ComposablePipeline`` — the
factory still wires the legacy class hierarchy. Phase 2.5 / Phase 3 will:

1. Write adapter classes that expose existing handler internals
   (``_run_llm_with_tools`` / ``_stream_tts_to_queue`` / ``LocalSTTInputMixin``)
   as the Protocols defined in :mod:`robot_comic.backends`.
2. Update ``handler_factory.py``'s ``composable`` branch to instantiate
   adapter-wrapped backends and inject them into a ``ComposablePipeline``.
3. Retire the legacy class hierarchy where this orchestrator covers it.

## Loop shape

For each completed user transcript:

1. Append ``{"role": "user", "content": transcript}`` to the conversation
   history.
2. Call ``llm.chat(history, tools)`` to get an :class:`LLMResponse`.
3. If the response has ``tool_calls``: dispatch each via the operator-
   supplied ``tool_dispatcher`` callback, append each result as a
   ``{"role": "tool", ...}`` history entry, and loop back to step 2.
   A per-turn ``max_tool_rounds`` cap prevents storms.
4. If the response has ``text``: append it as ``{"role": "assistant"}``
   and stream-synthesize it via ``tts.synthesize(text, tags)``, pushing
   each :class:`AudioFrame` into the operator-supplied output queue.

Tools are kept outside the LLM Protocol on purpose — the orchestrator
owns the dispatcher because tool implementations need access to
``ToolDependencies`` (robot, movement manager, camera worker, etc.),
which the LLM backend shouldn't see.

## Bundled-live backends

OpenAI Realtime / Gemini Live / HF Realtime fuse all three phases into
one session and do NOT compose into this orchestrator. The factory's
``PIPELINE_MODE`` dial (Phase 0) keeps them on their own path; this
class is only for composable-mode pipelines.
"""

from __future__ import annotations
import re
import time
import uuid
import asyncio
import difflib
import logging
import collections
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from opentelemetry import trace
from opentelemetry import context as otel_context

from robot_comic import telemetry
from robot_comic.pause import TranscriptDisposition
from robot_comic.config import set_custom_profile
from robot_comic.prompts import get_session_instructions
from robot_comic.backends import (
    ToolCall,
    AudioFrame,
    LLMBackend,
    STTBackend,
    TTSBackend,
    LLMResponse,
)
from robot_comic.history_trim import trim_history_in_place
from robot_comic.joke_history import record_joke_history
from robot_comic.welcome_gate import GateState, WelcomeGate
from robot_comic.tools.name_validation import record_user_transcript


if TYPE_CHECKING:
    from robot_comic.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


# Tool dispatcher callback. Receives a ToolCall (id + name + args) and returns
# the tool's result as a string that the orchestrator feeds back to the LLM
# as a ``{"role": "tool", "tool_call_id": ..., "content": ...}`` history entry.
ToolDispatcher = Callable[[ToolCall], Awaitable[str]]


# Default per-turn cap on consecutive tool round-trips. Matches the existing
# ``MAX_API_CALLS_PER_TURN`` in elevenlabs_tts.py (8) — Gemini 503 retries
# can otherwise burn the budget. See #286 / PR #322.
DEFAULT_MAX_TOOL_ROUNDS = 8


# Phase 5f.2 — minimum-utterance filter (self-echo cascade defence).
#
# Drops transcripts that look like speaker-echo / whisper-family
# hallucinations BEFORE they reach duplicate-suppression or the LLM.
# faster-whisper's webrtcvad-based segment dispatch is aggressive enough
# that room-acoustic reverb tail occasionally slips past the input-side
# echo guard; this orchestrator-side filter is the second line of defence
# and also helps Moonshine (which can occasionally hallucinate short
# partials too — see #19).
#
# Thresholds are heuristic; on-device tuning is a one-line change here.
# Rationale + the hardware finding that motivated the constants are in
# docs/superpowers/specs/2026-05-16-phase-5f-2-echo-cascade-fix.md.
MIN_TRANSCRIPT_WORDS = 2
MIN_TRANSCRIPT_CHARS = 8


# Phase 5f.3 — content-similarity echo filter.
#
# After 5f.2 closed the short-fragment arm of the cascade, hardware
# testing on 2026-05-16 showed longer multi-sentence echoes still
# cascading (faster-whisper transcribes the assistant's own speech back
# off the chassis speaker with only minor word errors; the transcript
# clears the 5f.2 length floor and dispatches as a fresh user turn).
#
# This filter compares each completed transcript against the last N
# assistant utterances; if any similarity ratio exceeds the threshold
# the transcript is dropped. Rationale + the hardware finding behind
# the constants live in
# ``docs/superpowers/specs/2026-05-16-phase-5f-3-echo-content-similarity-filter.md``.
ECHO_HISTORY_MAXLEN = 5
ECHO_SIMILARITY_THRESHOLD = 0.65


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize_for_echo_check(text: str) -> str:
    """Return a comparison-only form of ``text`` for echo similarity scoring.

    Lowercases, replaces every non-word non-whitespace character with a
    single space (so punctuation, apostrophes, ellipses all collapse),
    and squashes whitespace runs. The output is used only for the
    :func:`difflib.SequenceMatcher` ratio — the original transcript
    string is what would be dispatched / appended to history.
    """
    if not text:
        return ""
    lowered = text.lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", no_punct)
    return collapsed.strip()


class ComposablePipeline:
    """Orchestrates one STT + LLM + TTS pipeline via injected backends.

    Takes three backends (each satisfying its respective Protocol from
    :mod:`robot_comic.backends`) plus optional tool wiring and an output
    queue. The bound STT callback drives the main loop — every completed
    transcript runs one full user-turn cycle.

    Not thread-safe: assumes a single asyncio loop owns this instance.
    """

    def __init__(
        self,
        stt: STTBackend,
        llm: LLMBackend,
        tts: TTSBackend,
        *,
        output_queue: "asyncio.Queue[Any] | None" = None,
        tool_dispatcher: ToolDispatcher | None = None,
        tools_spec: list[dict[str, Any]] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        system_prompt: str | None = None,
        deps: "ToolDependencies | None" = None,
        welcome_gate: WelcomeGate | None = None,
    ) -> None:
        """Build the pipeline. See class docstring for arg semantics.

        Phase 5e.2 kwargs:

        - ``deps`` — when provided, the pipeline drives
          movement-manager listening state, head-wobbler reset,
          pause-controller integration, and
          name-validation transcript recording on each turn. Pre-5e.2
          callers that route those concerns through
          :class:`LocalSTTInputMixin` on the host pass ``None`` and the
          pipeline-side callbacks become safe no-ops.
        - ``welcome_gate`` — when provided, completed transcripts are
          checked against the gate before dispatch; in ``WAITING``
          state non-matching transcripts are dropped.
        """
        self.stt = stt
        self.llm = llm
        self.tts = tts
        # The queue is heterogeneous post-5e.2: TTS pushes ``AudioFrame``;
        # the orchestrator's transcript callbacks push
        # ``fastrtc.AdditionalOutputs`` envelopes for the admin UI.
        # Drainers (``ComposableConversationHandler.emit`` via
        # ``wait_for_item``) tolerate both.
        self.output_queue: "asyncio.Queue[Any]" = output_queue or asyncio.Queue()
        self.tool_dispatcher = tool_dispatcher
        self.tools_spec = tools_spec or []
        self.max_tool_rounds = max_tool_rounds
        self.deps = deps
        self.welcome_gate = welcome_gate

        # Barge-in flush callback (Phase 5e.2). The wrapper's
        # ``_clear_queue`` setter mirrors onto this attribute so the
        # ``_on_speech_started`` hook can flush the player when a new
        # user turn begins. ``None`` until ``LocalStream`` installs it.
        self._clear_queue: Callable[[], None] | None = None

        # Per-turn STT spans + duplicate-suppression state (Phase 5e.2,
        # mirrors ``LocalSTTInputMixin._handle_local_stt_event``).
        self._turn_span: Any = None
        self._turn_ctx_token: Any = None
        self._stt_infer_span: Any = None
        self._stt_infer_start: float = 0.0
        self._turn_id: str | None = None
        self._turn_start_at: float | None = None
        self._last_completed_transcript: str = ""
        self._last_completed_at: float = 0.0

        # Phase 5f.3 — ring buffer of recent assistant utterances
        # (normalized) for content-similarity echo filtering.
        self._recent_assistant_texts: collections.deque[str] = collections.deque(maxlen=ECHO_HISTORY_MAXLEN)

        self._conversation_history: list[dict[str, Any]] = []
        if system_prompt:
            self._conversation_history.append({"role": "system", "content": system_prompt})
        self._stop_event = asyncio.Event()
        self._started = False

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def start_up(self) -> None:
        """Prepare backends, bind the STT callback, and block until shutdown.

        Mirrors the existing ``ConversationHandler`` ABC's contract: this
        coroutine runs for the entire lifetime of the pipeline. Returns
        only when :meth:`shutdown` is called.
        """
        # If shutdown() already fired (stop_event is set), don't prepare any
        # backends — they'd never be torn down because shutdown's
        # ``if not self._started`` early-return has already run.
        if self._stop_event.is_set():
            return
        await self.llm.prepare()
        await self.tts.prepare()
        await self.stt.start(
            on_completed=self._on_transcript_completed,
            on_partial=self._on_partial_transcript,
            on_speech_started=self._on_speech_started,
        )
        self._started = True
        logger.info(
            "ComposablePipeline ready (stt=%s llm=%s tts=%s)",
            type(self.stt).__name__,
            type(self.llm).__name__,
            type(self.tts).__name__,
        )
        await self._stop_event.wait()

    async def shutdown(self) -> None:
        """Stop the STT pipeline, close LLM/TTS clients, release the loop."""
        if not self._started:
            self._stop_event.set()
            return
        try:
            await self.stt.stop()
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            logger.warning("STT.stop() raised: %s", exc)
        try:
            await self.llm.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM.shutdown() raised: %s", exc)
        try:
            await self.tts.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS.shutdown() raised: %s", exc)
        self._stop_event.set()

    # ---------------------------------------------------------------------
    # Audio input — operator pushes captured frames here.
    # ---------------------------------------------------------------------

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Forward a captured audio frame to the STT backend."""
        await self.stt.feed_audio(frame)

    # ---------------------------------------------------------------------
    # Conversation state
    # ---------------------------------------------------------------------

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Return the live conversation history (mutable)."""
        return self._conversation_history

    def reset_history(self, *, keep_system: bool = True) -> None:
        """Clear the conversation history. By default the system prompt persists."""
        if keep_system and self._conversation_history and self._conversation_history[0].get("role") == "system":
            self._conversation_history = [self._conversation_history[0]]
        else:
            self._conversation_history = []

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality, reset history + per-session TTS state, re-seed system prompt.

        Phase 5c.2 (#TBD) moved this body off
        :class:`ComposableConversationHandler` and onto the pipeline; the
        wrapper is now a thin pass-through. Persona switch is pipeline-shaped
        state surgery (history reset + system-prompt re-seed) plus a hard
        cut on TTS listening state via :meth:`TTSBackend.reset_per_session_state`,
        so the work belongs here rather than at the FastRTC-adapter
        boundary.

        Flow:

        1. :func:`set_custom_profile` — flip the global profile dial. On
           failure, return a "Failed to apply personality" string without
           touching history or per-session state (pins the partial-failure
           contract: a typo'd profile name does not half-reset the session).
        2. :meth:`reset_history` with ``keep_system=False`` — wipe
           everything, including the prior persona's system prompt.
        3. :meth:`TTSBackend.reset_per_session_state` — clear the wrapped
           handler's echo-guard accumulators (``_speaking_until``,
           ``_response_start_ts``, ``_response_audio_bytes``) so a stale
           playback deadline from the previous persona does not bleed
           into the new persona's listening window. Wired in Phase 5a.1
           on the wrapper, moved onto :class:`TTSBackend` in Phase 5c.2.
        4. Re-seed the system prompt with :func:`get_session_instructions`
           so the next LLM round-trip uses the new persona's instructions.

        Joke history is a cross-persona file
        (``~/.robot-comic/joke-history.json``) read in
        ``prompts.py:_append_joke_history`` — it intentionally persists
        across personality switches so each persona avoids the others'
        punchlines (legacy parity, see ``joke_history.format_for_prompt``
        comment "Pulls entries from ALL personas (cross-persona dedup)").
        """
        try:
            set_custom_profile(profile)
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"
        self.reset_history(keep_system=False)
        await self.tts.reset_per_session_state()
        # Phase 5f.3: drop the prior persona's echo-history so the new
        # persona's similarity filter starts clean.
        self._recent_assistant_texts.clear()
        self._conversation_history.append({"role": "system", "content": get_session_instructions()})
        return f"Applied personality {profile!r}. Conversation history reset."

    # ---------------------------------------------------------------------
    # Internal — transcript-triggered turn loop
    # ---------------------------------------------------------------------

    async def _on_speech_started(self) -> None:
        """STT speech-started callback (Phase 5e.2).

        Opens the root ``turn`` span + child ``stt.infer`` span, fires
        the barge-in flush callback if registered, resets the head
        wobbler, and signals the movement manager to enter "listening"
        state. All deps-dependent steps are guarded — pipelines
        constructed without ``deps`` (pre-5e.2 host-coupled triples)
        get the span open but skip the movement/wobbler hooks.

        Mirrors :class:`LocalSTTInputMixin._handle_local_stt_event`
        (``local_stt_realtime.py:518-562``) — the started-event arm.
        """
        # Close any leftover turn span from a previous (interrupted) turn.
        if self._turn_span is not None:
            try:
                self._turn_span.set_attribute("turn.outcome", "interrupted")
                self._turn_span.end()
            except Exception:  # pragma: no cover — best-effort span cleanup
                logger.debug("Failed to close prior turn span", exc_info=True)
            self._turn_span = None
        if self._stt_infer_span is not None:
            try:
                self._stt_infer_span.end()
            except Exception:  # pragma: no cover
                logger.debug("Failed to close prior stt.infer span", exc_info=True)
            self._stt_infer_span = None

        now = time.perf_counter()
        tracer = telemetry.get_tracer()
        self._turn_id = str(uuid.uuid4())
        self._turn_start_at = now
        self._turn_span = tracer.start_span(
            "turn",
            attributes={
                "turn.id": self._turn_id,
                "robot.mode": "local_stt",
                "robot.persona": telemetry.current_persona(),
            },
        )
        self._turn_ctx_token = otel_context.attach(trace.set_span_in_context(self._turn_span))
        self._stt_infer_span = tracer.start_span("stt.infer")
        self._stt_infer_start = now

        # Barge-in flush: dump any queued TTS frames from the prior turn so
        # the user's new utterance isn't preceded by stale audio.
        if self._clear_queue is not None:
            try:
                self._clear_queue()
            except Exception:  # pragma: no cover — flush is best-effort
                logger.debug("clear_queue raised in _on_speech_started", exc_info=True)

        if self.deps is not None:
            head_wobbler = getattr(self.deps, "head_wobbler", None)
            if head_wobbler is not None:
                try:
                    head_wobbler.reset()
                except Exception:  # pragma: no cover
                    logger.debug("head_wobbler.reset raised", exc_info=True)
            movement_manager = getattr(self.deps, "movement_manager", None)
            if movement_manager is not None:
                try:
                    movement_manager.set_listening(True)
                except Exception:  # pragma: no cover
                    logger.debug("movement_manager.set_listening(True) raised", exc_info=True)

    async def _on_partial_transcript(self, transcript: str) -> None:
        """STT partial-transcript callback (Phase 5e.2).

        Publishes the in-progress transcript to the output queue as a
        ``user_partial`` row so the admin UI live-transcript widget can
        render incremental updates. Mirrors
        :class:`LocalSTTInputMixin._handle_local_stt_event`
        (``local_stt_realtime.py:564-568``).
        """
        if not transcript:
            return
        from fastrtc import AdditionalOutputs  # deferred — fastrtc pulls gradio at boot

        await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))

    async def _on_transcript_completed(self, transcript: str) -> None:
        """Handle one completed user line: append to history, run LLM, speak.

        Phase 5e.2 extension: when ``deps`` is provided the pipeline
        also drives the per-turn closing rituals previously owned by
        :class:`LocalSTTInputMixin` — duplicate suppression, ``user``
        publication, ``set_listening(False)``,
        :func:`record_user_transcript` for tool-side name validation,
        pause-controller routing, welcome-gate gating, and ``stt.infer``
        span closure. Pre-5e.2 pipelines (``deps=None``) skip all of
        that and behave as before.
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return

        # Phase 5f.2 — minimum-utterance filter. Drops short transcripts
        # that almost always come from speaker-echo / whisper hallucination
        # (e.g. ``"You"``, ``"Thank you"``) before they reach the LLM and
        # restart the echo-cascade. Runs BEFORE duplicate-suppression so a
        # dropped fragment never poisons the dedupe cache. See spec
        # ``docs/superpowers/specs/2026-05-16-phase-5f-2-echo-cascade-fix.md``.
        if len(transcript.split()) < MIN_TRANSCRIPT_WORDS or len(transcript) < MIN_TRANSCRIPT_CHARS:
            logger.debug("dropping short transcript as likely echo: %r", transcript)
            return

        # Phase 5f.3 — content-similarity echo filter. Drops transcripts
        # that are substantially similar to one of the last N assistant
        # utterances (i.e. the chassis mic picked up our own speaker
        # output and faster-whisper transcribed it back, only mildly
        # corrupted). Runs BEFORE duplicate-suppression so the dedupe
        # cache is never poisoned with an echoed value. See spec
        # ``docs/superpowers/specs/2026-05-16-phase-5f-3-echo-content-similarity-filter.md``.
        if self._recent_assistant_texts:
            needle = _normalize_for_echo_check(transcript)
            if needle:
                best_ratio = max(
                    difflib.SequenceMatcher(None, needle, candidate).ratio()
                    for candidate in self._recent_assistant_texts
                )
                if best_ratio >= ECHO_SIMILARITY_THRESHOLD:
                    logger.debug(
                        "dropping likely echo of own speech: ratio=%.2f, transcript=%r",
                        best_ratio,
                        transcript[:60],
                    )
                    return

        # Duplicate-suppression window. Mirrors mixin lines 574-578.
        now = time.perf_counter()
        if transcript == self._last_completed_transcript and now - self._last_completed_at < 0.75:
            logger.debug("Ignoring duplicate transcript: %s", transcript)
            return
        self._last_completed_transcript = transcript
        self._last_completed_at = now

        if self.deps is not None:
            movement_manager = getattr(self.deps, "movement_manager", None)
            if movement_manager is not None:
                try:
                    movement_manager.set_listening(False)
                except Exception:  # pragma: no cover
                    logger.debug("movement_manager.set_listening(False) raised", exc_info=True)

            # Close stt.infer span with turn excerpt before LLM round-trip
            # so monitor consumers see the excerpt while the outer turn
            # span is still open.
            words = transcript.split()
            excerpt = " ".join(words[:5]) + ("…" if len(words) > 5 else "")
            stt_span = self._stt_infer_span
            if stt_span is not None:
                try:
                    stt_span.set_attribute("turn.excerpt", excerpt)
                    stt_span.end()
                    stt_s = now - self._stt_infer_start
                    telemetry.record_stt(
                        stt_s,
                        {"gen_ai.system": "local_stt", "stt.type": "moonshine"},
                    )
                except Exception:  # pragma: no cover
                    logger.debug("Failed to close stt.infer span", exc_info=True)
                self._stt_infer_span = None
            if self._turn_span is not None:
                try:
                    self._turn_span.set_attribute("turn.excerpt", excerpt)
                except Exception:  # pragma: no cover
                    logger.debug("Failed to tag turn span", exc_info=True)

            # Tool-side name-validation guard (#287).
            recent = getattr(self.deps, "recent_user_transcripts", None)
            if recent is not None:
                record_user_transcript(recent, transcript)

            # Publish completed transcript to the admin UI.
            from fastrtc import AdditionalOutputs  # deferred — fastrtc pulls gradio

            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            # Pause-controller routing. HANDLED short-circuits dispatch.
            pause_controller = getattr(self.deps, "pause_controller", None)
            if pause_controller is not None:
                try:
                    disposition = pause_controller.handle_transcript(transcript)
                except Exception as exc:
                    logger.error("pause_controller.handle_transcript raised: %s", exc)
                    disposition = TranscriptDisposition.DISPATCH
                if disposition is TranscriptDisposition.HANDLED:
                    return

        # Welcome-gate: drop WAITING transcripts that don't match the wake name.
        gate = self.welcome_gate
        if gate is not None and gate.state is GateState.WAITING:
            if not gate.consider(transcript):
                logger.debug("welcome gate: WAITING — transcript not dispatched: %r", transcript[:60])
                return
            # Gate just opened — dispatch this transcript normally.

        self._conversation_history.append({"role": "user", "content": transcript})
        try:
            await self._run_llm_loop_and_speak()
        except Exception:
            logger.exception("ComposablePipeline turn failed")

    async def _run_llm_loop_and_speak(self) -> None:
        """Run the LLM round-trips with tool dispatch, then synthesize speech."""
        # Lifecycle Hook #5 (#337): cap the conversation history at
        # ``REACHY_MINI_MAX_HISTORY_TURNS`` user turns so long sessions don't
        # blow the model's context window or run the token bill into the
        # ground. Once-per-user-turn cadence matches the legacy sites in
        # ``_dispatch_completed_transcript``. Legacy parity:
        # llama_base.py:506, gemini_tts.py:365, elevenlabs_tts.py:565.
        trim_history_in_place(self._conversation_history)
        for _round in range(self.max_tool_rounds):
            response = await self.llm.chat(
                self._conversation_history,
                tools=self.tools_spec or None,
            )
            if response.tool_calls:
                if self.tool_dispatcher is None:
                    logger.warning(
                        "LLM requested tools but no dispatcher is configured; "
                        "ignoring %d call(s) and breaking the loop",
                        len(response.tool_calls),
                    )
                    break
                await self._dispatch_tools_and_record(response.tool_calls)
                continue
            # No more tool calls — assistant text is final.
            await self._speak_assistant_text(response)
            return
        logger.warning(
            "ComposablePipeline hit max_tool_rounds=%d without an assistant text response",
            self.max_tool_rounds,
        )

    async def _dispatch_tools_and_record(self, tool_calls: tuple[ToolCall, ...]) -> None:
        """Dispatch each tool call sequentially and append results to history."""
        assert self.tool_dispatcher is not None  # caller guards
        for call in tool_calls:
            try:
                result = await self.tool_dispatcher(call)
            except Exception as exc:
                logger.exception("Tool %s raised", call.name)
                result = f"[tool error: {type(exc).__name__}: {exc}]"
            self._conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result,
                }
            )

    async def _speak_assistant_text(self, response: LLMResponse) -> None:
        """Append the assistant text to history and stream TTS frames out."""
        text = response.text
        if not text.strip():
            logger.warning("Assistant returned empty text; nothing to speak")
            return
        # Lifecycle Hook #4 (#337): capture the punchline + topic into joke
        # history so the next session's system-prompt builder can include
        # them in the "RECENT JOKES (DO NOT REPEAT)" section. Best-effort —
        # the helper swallows its own exceptions; the outer ``try`` is
        # belt-and-braces so a future code change in the helper can't kill
        # the turn before TTS runs. Legacy parity sites:
        # llama_base.py:578-594 and gemini_tts.py:380-394.
        try:
            await record_joke_history(text)
        except Exception as exc:  # pragma: no cover — helper is itself defensive
            logger.debug("record_joke_history raised through to orchestrator: %s", exc)
        self._conversation_history.append({"role": "assistant", "content": text})
        # Phase 5f.3 — record the normalized assistant text in the ring
        # buffer so the next user transcript's similarity check has the
        # latest utterance to compare against. Skip empty normalized
        # output (would dominate the SequenceMatcher ratio with garbage).
        normalized = _normalize_for_echo_check(text)
        if normalized:
            self._recent_assistant_texts.append(normalized)
        # Phase 5a.2: thread the structured ``delivery_tags`` channel from
        # ``LLMResponse`` to ``TTSBackend.synthesize(tags=...)``. Today's
        # LLM adapters leave the tuple empty, so TTS adapters fall back to
        # the today text-parsing behaviour; future PRs that surface
        # structured cues from the LLM populate the field and the consume
        # path activates without further orchestrator changes.
        # ``first_audio_marker`` (Phase 5a.2): allocate a fresh list per
        # turn so the TTS adapter can stamp first-audio-out wallclock;
        # downstream telemetry consumers are a separate PR. See
        # ``docs/superpowers/specs/2026-05-16-phase-5a2-delivery-tag-plumbing.md``.
        first_audio_marker: list[float] = []
        async for frame in self.tts.synthesize(
            text,
            tags=response.delivery_tags,
            first_audio_marker=first_audio_marker,
        ):
            # ``console.py``'s playback loop branches on
            # ``isinstance(handler_output, tuple)`` to dispatch to ALSA
            # (see ``console.py:1466``). The TTS adapters yield
            # :class:`AudioFrame` dataclasses for the Protocol-level
            # surface; the legacy ``elevenlabs_tts.py`` put-site (line
            # ~487) bypassed the wrapping and pushed
            # ``(sample_rate, ndarray)`` tuples straight onto the queue.
            # Unwrap here so downstream consumers stay shape-compatible.
            # Without this, TTS audio is silently dropped on hardware
            # (the dataclass falls through ``isinstance(..., tuple)`` and
            # ``isinstance(..., AdditionalOutputs)`` checks). Caught
            # during 2026-05-16 hardware validation on ricci.
            await self.output_queue.put((frame.sample_rate, frame.samples))
