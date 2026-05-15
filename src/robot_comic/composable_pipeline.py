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
import asyncio
import logging
from typing import Any, Callable, Awaitable

from robot_comic.backends import (
    ToolCall,
    AudioFrame,
    LLMBackend,
    STTBackend,
    TTSBackend,
    LLMResponse,
)


logger = logging.getLogger(__name__)


# Tool dispatcher callback. Receives a ToolCall (id + name + args) and returns
# the tool's result as a string that the orchestrator feeds back to the LLM
# as a ``{"role": "tool", "tool_call_id": ..., "content": ...}`` history entry.
ToolDispatcher = Callable[[ToolCall], Awaitable[str]]


# Default per-turn cap on consecutive tool round-trips. Matches the existing
# ``MAX_API_CALLS_PER_TURN`` in elevenlabs_tts.py (8) — Gemini 503 retries
# can otherwise burn the budget. See #286 / PR #322.
DEFAULT_MAX_TOOL_ROUNDS = 8


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
        output_queue: asyncio.Queue[AudioFrame] | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
        tools_spec: list[dict[str, Any]] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        system_prompt: str | None = None,
    ) -> None:
        """Build the pipeline. See class docstring for arg semantics."""
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.output_queue: asyncio.Queue[AudioFrame] = output_queue or asyncio.Queue()
        self.tool_dispatcher = tool_dispatcher
        self.tools_spec = tools_spec or []
        self.max_tool_rounds = max_tool_rounds

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
        await self.llm.prepare()
        await self.tts.prepare()
        await self.stt.start(on_completed=self._on_transcript_completed)
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

    # ---------------------------------------------------------------------
    # Internal — transcript-triggered turn loop
    # ---------------------------------------------------------------------

    async def _on_transcript_completed(self, transcript: str) -> None:
        """Handle one completed user line: append to history, run LLM, speak."""
        if not transcript:
            return
        self._conversation_history.append({"role": "user", "content": transcript})
        try:
            await self._run_llm_loop_and_speak()
        except Exception:
            logger.exception("ComposablePipeline turn failed")

    async def _run_llm_loop_and_speak(self) -> None:
        """Run the LLM round-trips with tool dispatch, then synthesize speech."""
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
        self._conversation_history.append({"role": "assistant", "content": text})
        async for frame in self.tts.synthesize(text):
            await self.output_queue.put(frame)
