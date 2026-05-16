"""GeminiTTSAdapter: expose GeminiTTSResponseHandler's TTS half as TTSBackend.

The legacy ``GeminiTTSResponseHandler._call_tts_with_retry(text,
system_instruction=None)`` issues a single Gemini TTS call and returns raw
16-bit PCM ``bytes``. The Phase 1 ``TTSBackend.synthesize`` Protocol yields
:class:`AudioFrame` objects one chunk at a time. The adapter replicates the
legacy per-sentence loop from
``GeminiTTSResponseHandler._dispatch_completed_transcript`` (``gemini_tts.py``,
lines ~396-418) so the wrapper produces the same audio shape as the legacy
handler does today:

1. ``split_sentences(text)`` — break the LLM output on sentence boundaries.
2. For each sentence:
   a. ``strip_gemini_tags`` removes ``[fast]`` / ``[annoyance]`` /
      ``[short pause]`` markers from the spoken text.
   b. ``extract_delivery_tags`` reads the markers back out as a tag list.
   c. If ``[short pause]`` was present, yield :class:`AudioFrame` chunks of
      silence before the spoken audio (``SHORT_PAUSE_MS`` at the TTS sample
      rate).
   d. ``build_tts_system_instruction`` appends a delivery cue suffix to the
      base persona instruction.
   e. ``_call_tts_with_retry`` runs the actual Gemini TTS call with the
      tag-augmented instruction; returns raw PCM bytes (or ``None`` on
      failure → skip).
   f. ``_pcm_to_frames`` chunks the bytes into 100 ms frames; one
      :class:`AudioFrame` is yielded per chunk.

There is no temp-queue swap (unlike :class:`ElevenLabsTTSAdapter` and
:class:`ChatterboxTTSAdapter`) because ``_call_tts_with_retry`` returns the
PCM payload directly rather than pushing it to ``self.output_queue``. The
adapter chunks inline and yields without any streaming-task lifecycle.

## Bundled-handler pairing

This adapter wraps a :class:`GeminiTTSResponseHandler` — a class that
fuses LLM + TTS over one ``genai.Client`` instance. The companion adapter
:class:`~robot_comic.adapters.gemini_bundled_llm_adapter.GeminiBundledLLMAdapter`
wraps the LLM half of the SAME handler instance, so a single ``genai.Client``
backs both Protocol surfaces — see ``docs/superpowers/specs/2026-05-15-phase-4c5-gemini-tts-adapter.md``
Q1 for the design rationale (option (b): two adapters on one handler).

## Tag handling — consume-or-fallback (Phase 5a.2)

The Protocol's ``tags: tuple[str, ...]`` parameter is now honoured with a
fallback path so the legacy text-embedded markers keep working:

- ``tags`` non-empty → the adapter uses those tags as the per-call delivery
  cue for every sentence. ``SHORT_PAUSE_TAG`` in the param emits a single
  leading silence burst (not per-sentence — the param applies to the whole
  ``synthesize()`` call).
- ``tags`` empty → fall back to per-sentence
  :func:`extract_delivery_tags(sentence)`. ``[short pause]`` triggers
  per-occurrence silence frames — today's behaviour, unchanged.

LLM adapters today leave ``LLMResponse.delivery_tags`` empty (the producer
side is a separate concern); the fallback path therefore stays the
production-hot path. Future PRs that surface structured delivery cues from
the LLM populate ``delivery_tags`` and the consume path activates.

## ``shutdown()`` is a no-op

``genai.Client`` has no explicit ``aclose`` method as used by
``GeminiTTSResponseHandler``. The factory's composable host drains its
own ``output_queue`` — not relevant on the adapter side because the
adapter doesn't own a queue.
"""

from __future__ import annotations
import time
import asyncio
import logging
from typing import Any, Protocol, AsyncIterator

from robot_comic.backends import AudioFrame
from robot_comic.gemini_tts import (
    SHORT_PAUSE_MS,
    SHORT_PAUSE_TAG,
    GEMINI_TTS_OUTPUT_SAMPLE_RATE,
    GeminiTTSResponseHandler,
    _silence_pcm,
    extract_delivery_tags,
    build_tts_system_instruction,
    load_profile_tts_instruction,
)
from robot_comic.llama_base import split_sentences
from robot_comic.chatterbox_tag_translator import strip_gemini_tags


logger = logging.getLogger(__name__)


class _GeminiTTSCompatibleHandler(Protocol):
    """Duck-typed surface ``GeminiTTSAdapter`` and ``GeminiBundledLLMAdapter`` share.

    Captures the members both adapters touch on the wrapped handler:

    - ``_prepare_startup_credentials()`` — awaited from ``prepare()``.
    - ``_call_tts_with_retry(text, system_instruction=...)`` — TTS call;
      returns raw PCM ``bytes`` or ``None``.
    - ``_run_llm_with_tools()`` — LLM call returning final assistant text;
      used by :class:`GeminiBundledLLMAdapter`, declared here so both
      adapters can share the same Protocol type.
    - ``_conversation_history`` — list[dict[str, Any]] — swapped by the LLM
      adapter for the call's duration.
    - ``_client`` — opaque (``genai.Client | None``). The adapters never
      access it directly, but ``_run_llm_with_tools`` / ``_call_tts_with_retry``
      both ``assert self._client is not None`` internally — declared here so
      mypy / Protocol structural matching sees the attribute.

    Concrete satisfier today: :class:`GeminiTTSResponseHandler` (the
    factory composes :class:`LocalSTTInputMixin` over it for the live
    pipeline).

    Not ``@runtime_checkable`` — we don't use ``isinstance(handler,
    _GeminiTTSCompatibleHandler)`` anywhere; mypy structural matching is
    the only contract surface.
    """

    _conversation_history: list[dict[str, Any]]
    _client: Any

    async def _prepare_startup_credentials(self) -> None: ...

    async def _call_tts_with_retry(
        self,
        text: str,
        system_instruction: str | None = None,
    ) -> bytes | None: ...

    async def _run_llm_with_tools(self) -> str: ...


class GeminiTTSAdapter:
    """Adapter exposing ``GeminiTTSResponseHandler`` as ``TTSBackend``."""

    def __init__(self, handler: "_GeminiTTSCompatibleHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's ``genai.Client``.

        Idempotent on the wrapped handler — the legacy
        ``_prepare_startup_credentials`` reassigns ``self._client`` to a fresh
        instance on every call. Same double-init pattern as 4c.3 flagged when
        ``GeminiLLMAdapter`` and ``ElevenLabsTTSAdapter`` share a handler;
        out of scope here.
        """
        await self._handler._prepare_startup_credentials()

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        """Stream PCM frames for *text* as :class:`AudioFrame` instances.

        Replicates the per-sentence loop from
        ``GeminiTTSResponseHandler._dispatch_completed_transcript`` so the
        wrapper produces the same audio as the legacy handler does today.

        Phase 5a.2 — tag handling (consume-or-fallback):

        - When ``tags`` is **empty** (the orchestrator default), the adapter
          falls back to per-sentence text parsing via
          :func:`extract_delivery_tags`. ``[short pause]`` triggers a
          pre-sentence silence frame per occurrence (today's behaviour).
        - When ``tags`` is **non-empty**, those tags are used as the
          per-call delivery cue for every sentence's
          :func:`build_tts_system_instruction`. ``SHORT_PAUSE_TAG`` in the
          param emits silence **once** at the head of the stream (not
          per-sentence — the param applies to the whole call).

        Phase 5a.2 — ``first_audio_marker`` (when non-None) receives a
        single ``time.monotonic()`` append on the first yielded frame so
        the orchestrator can record per-turn first-audio latency.

        Boot-timeline emit: each frame yields a call to
        ``telemetry.emit_first_greeting_audio_once`` before the ``yield`` so
        the composable ``(moonshine, gemini_tts)`` triple lights the monitor's
        ``first_greeting.tts_first_audio`` row at the same point the legacy
        ``_dispatch_completed_transcript`` does (``gemini_tts.py``). The
        helper has a process-level once-guard, so the per-frame call only
        fires the emit once; subsequent calls short-circuit cheaply. See
        Lifecycle Hook #3b spec (``docs/superpowers/specs/2026-05-16-lifecycle-hook-3b-gemini-tts-first-greeting.md``).
        """
        if not text:
            return

        base_instruction = load_profile_tts_instruction()
        sentences = split_sentences(text) or [text]
        # Phase 5a.2 consume-or-fallback: when ``tags`` is non-empty, use it
        # as the per-call delivery cue and skip per-sentence text parsing.
        param_tags: list[str] | None = list(tags) if tags else None
        # When the param drives the cue, ``[short pause]`` emits a single
        # leading silence burst. The fallback path keeps per-sentence
        # silence semantics so multi-sentence text with embedded
        # ``[short pause]`` markers behaves as before.
        if param_tags is not None and SHORT_PAUSE_TAG in param_tags:
            for frame in GeminiTTSResponseHandler._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS)):
                yield AudioFrame(samples=frame, sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE)
        _marker_appended = False
        for sentence in sentences:
            spoken = strip_gemini_tags(sentence)
            if not spoken:
                continue
            if param_tags is not None:
                sentence_tags = param_tags
            else:
                sentence_tags = extract_delivery_tags(sentence)
                if SHORT_PAUSE_TAG in sentence_tags:
                    for frame in GeminiTTSResponseHandler._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS)):
                        yield AudioFrame(samples=frame, sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE)
            instruction = build_tts_system_instruction(base_instruction, sentence_tags)
            try:
                pcm_bytes = await self._handler._call_tts_with_retry(spoken, system_instruction=instruction)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "GeminiTTSAdapter: TTS call raised for sentence %r: %s",
                    spoken[:60],
                    exc,
                )
                continue
            if pcm_bytes is None:
                continue
            for frame in GeminiTTSResponseHandler._pcm_to_frames(pcm_bytes):
                # Boot-timeline event (#321 / Lifecycle Hook #3b): the helper
                # owns the once-per-process dedupe, so calling per frame is
                # safe and matches the legacy emit site at
                # ``gemini_tts.py:415``. Imported lazily to mirror the legacy
                # pattern and keep cold-import cost off this hot loop.
                from robot_comic import telemetry as _telemetry

                _telemetry.emit_first_greeting_audio_once()
                if first_audio_marker is not None and not _marker_appended:
                    first_audio_marker.append(time.monotonic())
                    _marker_appended = True
                yield AudioFrame(samples=frame, sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE)

    async def shutdown(self) -> None:
        """No-op — ``genai.Client`` has no explicit close path here."""
        return None
