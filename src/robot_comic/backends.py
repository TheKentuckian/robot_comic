"""STT / LLM / TTS backend Protocols — Phase 1 of the pipeline refactor.

Defines the three interfaces that the composable conversation handler
composes via dependency injection. The legacy handler classes expressed
their 3-phase pipeline via inheritance — every ``(STT, LLM, TTS)``
triple was a distinct class, with the LLM dial baked into the class
hierarchy (``LlamaElevenLabsTTSResponseHandler`` vs
``GeminiTextElevenLabsResponseHandler``).

Once Phase 2 lands, swapping the LLM means swapping one object, not
introducing a new subclass. These Protocols pin the contract the existing
handler classes need to satisfy (via thin adapters) for that to work.

**Phase 1 scope (this PR)**: define the contracts only. No production handler
class is changed. Tests cover the Protocols themselves and provide reference
in-memory implementations.

**Phase 2 (next PR)**: write adapter mixins / classes that surface each
existing handler's ``_run_llm_with_tools`` / ``_stream_tts_to_queue`` etc. as
the Protocol methods declared here. Introduce a composable
``ConversationHandler`` that takes injected backends. Update the factory's
composable-mode branch to use it.

**Phase 3**: retire the legacy class hierarchy where Phase 2's DI handler
covers it.

## Scope boundaries

These Protocols only model the composable pipeline. Bundled
speech-to-speech backends (OpenAI Realtime, Gemini Live, HF Realtime) fuse
all three phases into one session and don't decompose cleanly into separate
STT/LLM/TTS objects. The factory's ``PIPELINE_MODE`` dial (Phase 0) keeps
those backends on their own path; they're untouched by this refactor.
"""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Protocol,
    Awaitable,
    AsyncIterator,
    runtime_checkable,
)
from dataclasses import field, dataclass


# ---------------------------------------------------------------------------
# Shared data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """An LLM-requested tool invocation.

    ``id`` is a per-call identifier the LLM provides so the assistant can
    refer back to the call when the result comes in. ``args`` is the parsed
    JSON arguments object — the orchestrator hands these to the actual tool
    implementation. The result of dispatch goes back to the LLM as a
    ``tool``-role message in the next chat turn.
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """One round-trip output from :meth:`LLMBackend.chat`.

    Either ``text`` is set (an assistant message ready for TTS) or
    ``tool_calls`` is non-empty (the LLM wants tools dispatched and another
    round-trip). Implementations MAY populate both — the orchestrator
    decides what to do based on which is present.

    ``delivery_tags`` carries optional delivery hints (``fast``, ``slow``,
    ``annoyance``, ``short pause``, etc.) that downstream TTS backends may
    consume in lieu of parsing the text. When empty (the default), TTS
    adapters fall back to text-based extraction — the today behaviour where
    LLM prompts embed ``[fast]``-style markers in the spoken text. The
    orchestrator (``ComposablePipeline._speak_assistant_text``) passes this
    tuple through to ``TTSBackend.synthesize(tags=...)``.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    # Free-form provider-specific metadata (input_tokens, output_tokens,
    # finish_reason, etc.). Kept opaque so the Protocol doesn't pin a
    # specific telemetry shape.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Structured delivery hints for the TTS layer. Empty = "fall back to
    # text-based tag extraction" (today's behaviour for GeminiTTSAdapter
    # and ChatterboxTTSAdapter); non-empty = "use these as the per-call
    # delivery cue, ignore any markers embedded in the text".
    delivery_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AudioFrame:
    """A chunk of PCM audio produced by a TTS backend.

    ``samples`` is the raw audio data — typed as ``Any`` here so the Protocol
    doesn't force numpy as a dependency at the type level. In practice it's a
    ``numpy.ndarray[Any, np.dtype[np.int16]]`` matching the existing handler
    conventions (24 kHz, mono).
    """

    samples: Any
    sample_rate: int


# Transcript callback: a coroutine the orchestrator registers with the STT
# backend. Fires once per completed user line.
TranscriptCallback = Callable[[str], Awaitable[None]]


# ---------------------------------------------------------------------------
# Backend Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class STTBackend(Protocol):
    """Speech-to-text input backend.

    Lifecycle:
    1. ``start(on_completed)`` — register the transcript callback and bind
       to the audio input device / stream.
    2. ``feed_audio(frame)`` — called repeatedly by the audio pipeline.
    3. ``stop()`` — release the device / stream.

    The orchestrator owns the audio pipeline and calls ``feed_audio`` for
    every captured frame; the backend's job is to surface completed
    transcripts via the callback.
    """

    async def start(self, on_completed: TranscriptCallback) -> None:
        """Bind the transcript callback and prepare to receive audio."""
        ...

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Push one captured audio frame into the recognizer."""
        ...

    async def stop(self) -> None:
        """Release any resources held by the STT pipeline."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Large-language-model backend (text → text + optional tool calls).

    Lifecycle:
    1. ``prepare()`` — initialise client / credentials / connection pool.
    2. ``chat(messages, tools)`` — one round-trip. Called repeatedly by the
       orchestrator: once for a user turn, then again after each tool
       dispatch if the LLM keeps requesting tools.
    3. ``shutdown()`` — release HTTP / websocket clients.

    History management is the orchestrator's responsibility — the backend
    receives the full ``messages`` list every call and is stateless between
    invocations.
    """

    async def prepare(self) -> None:
        """Initialise client(s). Safe to call multiple times."""
        ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run one chat round-trip and return the response."""
        ...

    async def shutdown(self) -> None:
        """Close client(s)."""
        ...


@runtime_checkable
class TTSBackend(Protocol):
    """Text-to-speech output backend.

    Lifecycle:
    1. ``prepare()`` — initialise client / API session.
    2. ``synthesize(text, tags, first_audio_marker)`` — produce an async
       stream of PCM frames.
       ``tags`` carries optional delivery hints (``fast``, ``annoyance``,
       ``slow``, etc.) that some backends map to voice-settings deltas or
       per-call delivery cues. Adapters that today parse hints out of the
       text fall back to that path when ``tags`` is empty; non-empty
       ``tags`` overrides text-parsing as the structured channel.
       ``first_audio_marker`` (Phase 5a.2): when provided, the adapter
       appends ``time.monotonic()`` to the list on the first PCM frame it
       yields. Single-shot per call — adapters guard with a local flag so
       repeated frames don't append. Enables the orchestrator to record
       per-turn first-audio latency without subscribing to internal
       events. ``None`` (the default) opts out of the channel.
    3. ``shutdown()`` — release HTTP / websocket clients.

    The orchestrator owns the output queue; the backend simply yields
    frames in arrival order. Backpressure (queue full) is handled outside
    the backend.
    """

    async def prepare(self) -> None:
        """Initialise client(s). Safe to call multiple times."""
        ...

    def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        """Stream synthesised PCM frames for *text*.

        Implementations are typically async generators
        (``async def synthesize(...): ... yield AudioFrame(...)``); calling
        an async-generator function returns the iterator directly without
        ``await``, which matches the plain-``def`` declaration here. A
        synchronous function returning an ``AsyncIterator`` would also
        satisfy the Protocol — ``runtime_checkable`` checks method names
        only, not signatures.
        """
        ...

    async def shutdown(self) -> None:
        """Close client(s)."""
        ...

    async def get_available_voices(self) -> list[str]:
        """List voice IDs this backend can switch to.

        Default raises :class:`NotImplementedError`. The legacy
        TTS-holding handlers (ElevenLabs / Chatterbox / Gemini TTS) all
        implement this; the adapters forward through. Pure TTS backends
        without a voice concept (future Whisper-only or pipeline-debug
        TTS stubs) inherit the default.

        ``@runtime_checkable`` treats this as a required attribute for
        ``isinstance`` checks even though it's defaulted, so test stubs
        that satisfy :class:`TTSBackend` structurally (no inheritance)
        must declare the method. Stubs that inherit from the Protocol
        inherit the default raiser.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support voice listing")

    def get_current_voice(self) -> str:
        """Return the active voice ID.

        Sync — admin-UI call sites consume the value without an event
        loop. Default raises :class:`NotImplementedError`; see
        :meth:`get_available_voices` for the inheritance-vs-structural
        caveat.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support voice queries")

    async def change_voice(self, voice: str) -> str:
        """Switch the active voice; return a confirmation string.

        The legacy contract is lenient: unknown voices are stored as-is
        and resolved at synthesis time (or the synthesis path falls back
        to a default). Adapters preserve that behavior — see
        ``tests/adapters/test_tts_backend_contract.py::test_change_voice_to_unknown_voice_does_not_raise``.
        Default raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support voice switching")

    async def reset_per_session_state(self) -> None:
        """Clear per-session state on the backend (Phase 5c.2).

        Called by :meth:`ComposablePipeline.apply_personality` so persona
        switch is a hard cut on listening state. Adapters that wrap a
        legacy handler with echo-guard accumulators (``_speaking_until``,
        ``_response_start_ts``, ``_response_audio_bytes`` — see
        ``elevenlabs_tts.py:558-560`` per-turn reset and
        ``llama_base.py:182-184`` canned-opener reset for legacy parity
        sites) zero them here so a stale playback deadline from the
        previous persona does not bleed into the next persona's
        listening window.

        Default is a clean no-op — backends without per-session state
        (Gemini TTS today, future pipeline-debug stubs) inherit it for
        free. Unlike :meth:`get_available_voices` /
        :meth:`get_current_voice` / :meth:`change_voice` which raise
        :class:`NotImplementedError` (operator-facing surface where a
        silent no-op would be confusing), this is an internal cleanup
        hook the orchestrator fires opportunistically — a no-op default
        is correct.

        ``@runtime_checkable`` treats this as a required attribute for
        ``isinstance`` checks even when defaulted; structural-only
        satisfiers (test stubs that don't inherit from the Protocol)
        must declare the method.
        """
        return None
