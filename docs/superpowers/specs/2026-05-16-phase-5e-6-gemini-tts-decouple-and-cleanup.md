# Phase 5e.6 — Migrate `(moonshine, gemini-bundled, gemini_tts)` off `LocalSTTInputMixin` + 5e cleanup

**Date:** 2026-05-16
**Status:** Spec — implementation on
`claude/phase-5e-6-gemini-tts-decouple-and-cleanup`.
**Tracks:** epic #391; established pattern from
`docs/superpowers/specs/2026-05-16-phase-5e-2-llama-elevenlabs-decouple.md`
§4 and the "5e.6" paragraph of §4.
**Predecessor:** Phase 5e.5 (`#412`, migrated `(moonshine, gemini,
elevenlabs)` off the mixin). All five composable triples — except
`(moonshine, gemini-bundled, gemini_tts)` — are now off
:class:`LocalSTTInputMixin`. The last factory-private mixin host
(``_LocalSTTGeminiTTSHost``) and the legacy ``_clear_queue`` mirror
branch on the wrapper survive only because this PR retires them.

---

## §1 — Scope

This PR closes out the Phase 5e mixin-retirement arc by doing two
things together:

**Part A — Migrate the final triple off the mixin.** Mechanical mirror
of 5e.2 / 5e.3 / 5e.4 / 5e.5 for the `(moonshine, gemini-bundled,
gemini_tts)` triple. **No new design decisions.** The Protocol,
`ComposablePipeline` host-concern landing, `MoonshineSTTAdapter`
standalone wiring, `_clear_queue` double-mirror, and the
`_maybe_build_welcome_gate` factory helper are all already in place
from 5e.2 and unchanged here.

**Part B — 5e cleanup.** With every composable triple now using the
new shape, the transition scaffolding can come down:

1. The `_clear_queue` legacy `_tts_handler` mirror branch on
   :class:`ComposableConversationHandler._clear_queue.setter` is dead
   code — no migrated triple's `LocalSTTInputMixin` reads the
   callback off the host shell anymore.
2. The ``LocalSTTInputMixin`` class docstring is updated to mark it as
   hybrid-only — it survives as a base class for the two
   ``LocalSTT*RealtimeHandler`` hybrids (per the Phase 4c-tris Option
   B "legacy forever" decision) but is no longer used by any
   composable triple.
3. `PIPELINE_REFACTOR.md` Phase 5 status table records 5e complete.

### §1.1 — What 5e.6 does NOT do

- **Does not** delete `LocalSTTInputMixin`. It still serves the two
  `LocalSTT*RealtimeHandler` hybrids in `local_stt_realtime.py`. Per
  the Phase 4c-tris decision those hybrids are legacy-forever; the
  mixin stays.
- **Does not** touch `LocalSTTOpenAIRealtimeHandler` /
  `LocalSTTHuggingFaceRealtimeHandler`. They remain
  `LocalSTTInputMixin`-based unchanged.
- **Does not** extend the `STTBackend` Protocol or change
  `ComposablePipeline` / `MoonshineSTTAdapter`. The 5e.2 pattern is
  proven and reused verbatim.
- **Does not** add faster-whisper or any new STT backend. That's 5f.
- **Does not** drop `self._tts_handler` from the wrapper. See §2.3
  for the audit findings — factory tests across multiple files use
  `result._tts_handler` to confirm the right concrete handler was
  composed, so the field stays.

## §2 — Part A — `gemini_tts` triple migration

### §2.1 — `_prepare_startup_credentials` guard placement

`GeminiTTSResponseHandler._prepare_startup_credentials`
(`src/robot_comic/gemini_tts.py:242-253`) is a short method: it only
builds a single `genai.Client` and logs once. There is no `super()`
call (the handler inherits from `AsyncStreamHandler` only — no
`*ResponseHandler` cooperative chain). The leaf-only guard wraps the
whole body so the second invocation is a cheap no-op:

```python
async def _prepare_startup_credentials(self) -> None:
    if getattr(self, "_startup_credentials_ready", False):
        return
    from google import genai  # deferred: google.genai.types costs ~5.5 s at boot

    api_key = config.GEMINI_API_KEY or "DUMMY"
    self._client = genai.Client(api_key=api_key)
    logger.info(
        "GeminiTTS handler initialised: llm=%s tts=%s voice=%s",
        GEMINI_TTS_LLM_MODEL,
        GEMINI_TTS_MODEL,
        self.get_current_voice(),
    )
    self._startup_credentials_ready = True
```

The flag is set only on the success path so a failed run re-attempts
the full chain. Matches 5e.2 / 5e.3 / 5e.4 / 5e.5 contract verbatim.

Unlike the gemini-text leaves, this handler has only one adapter
chain (the bundled LLM and TTS adapters wrap the same handler and
call its `_prepare_startup_credentials` once each — see
`GeminiBundledLLMAdapter.prepare` and `GeminiTTSAdapter.prepare`), so
the duplicate-call risk is small but non-zero. The guard is for the
same defensive reason as the 5e.3-5e.5 leaves: post-migration the
plain handler is called by the two adapter `prepare` chains
independently, and without a guard each duplicate call would leak a
fresh `genai.Client`.

### §2.2 — Factory builder rewrite

Identical shape to 5e.5's `_build_composable_gemini_elevenlabs`. The
only triple-specific substitutions are the LLM adapter
(`GeminiBundledLLMAdapter` — bundled-Gemini, exposes
`_run_llm_with_tools` rather than `_call_llm`, see Phase 4c.5 spec)
and the TTS adapter (`GeminiTTSAdapter`).

`GeminiTTSResponseHandler` does not inherit from any TTS base that
sets `_speaking_until`. The handler-private TTS audio loop writes
`output_queue` directly without an echo-guard deadline. Today the
mixin's `receive` echo-guard (`local_stt_realtime.py:796-798`) reads
`self._speaking_until` off the host shell — but the handler itself
doesn't define `_speaking_until`. So when the mixin shell is removed,
the `should_drop_frame` closure reading
`getattr(host, "_speaking_until", 0.0)` will always return `0.0` →
the echo-guard is effectively a no-op for this triple, matching the
pre-5e.6 behaviour (the host shell inherited the `_speaking_until`
attribute only through the mixin's `__init__` not setting it; the
attribute existed but stayed at its default).

Wiring `should_drop_frame` uniformly across all five triples (rather
than special-casing this one as `None`) keeps the factory builders
mechanically identical and avoids a "why is this triple different"
footgun later. The closure is a cheap no-op when the handler never
writes `_speaking_until`.

### §2.3 — Builder shape

```python
def _build_composable_gemini_tts(**handler_kwargs: Any) -> Any:
    import time as _time

    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiTTSAdapter,
        MoonshineSTTAdapter,
        GeminiBundledLLMAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        host = GeminiTTSResponseHandler(**handler_kwargs)

        def _should_drop_frame() -> bool:
            # ``GeminiTTSResponseHandler`` does not write
            # ``_speaking_until``; getattr default keeps this a no-op
            # in practice, matching pre-5e.6 behaviour where the
            # mixin host inherited the attribute at its default value.
            return _time.perf_counter() < getattr(host, "_speaking_until", 0.0)

        stt = MoonshineSTTAdapter(should_drop_frame=_should_drop_frame)
        llm = GeminiBundledLLMAdapter(host)
        tts = GeminiTTSAdapter(host)
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
```

`_LocalSTTGeminiTTSHost` is deleted — this PR is its only call site
and it was the last factory-private mixin host.

## §3 — Part B — Cleanup

### §3.1 — Remove `_clear_queue` legacy `_tts_handler` mirror branch

`ComposableConversationHandler._clear_queue.setter`
(`src/robot_comic/composable_conversation_handler.py:89-114`) today
double-mirrors the callback onto both the pipeline (for migrated
triples) AND the wrapped TTS handler (for un-migrated triples that
still ran the mixin's `LocalSTTInputMixin._open_local_stt_stream` →
`self._clear_queue` barge-in path on the host shell).

Post-5e.6 every triple is migrated; the `_tts_handler` mirror branch
serves no live reader. Drop the branch:

```python
@_clear_queue.setter
def _clear_queue(self, callback: Callable[[], None] | None) -> None:
    self.__clear_queue = callback
    if getattr(self, "pipeline", None) is not None:
        self.pipeline._clear_queue = callback
```

The two existing tests (`test_clear_queue_assignment_propagates_to_tts_handler`,
`test_clear_queue_assignment_handles_none`) that assert
`wrapper._tts_handler._clear_queue is cb` are stale — they pin the
legacy double-mirror that was always intended to retire at 5e.6.
They get updated to:

- Drop the `_tts_handler._clear_queue` assertion; the surviving
  pipeline-mirror tests (`test_clear_queue_assignment_also_mirrors_onto_pipeline`,
  `test_clear_queue_assignment_mirrors_none_onto_pipeline`) cover
  the live behaviour.
- Or rename + simplify into a single test asserting the pipeline
  mirror.

Chosen approach: **delete** the two `_tts_handler`-mirror tests, keep
the pipeline-mirror pair as the live contract. This matches the spec
intent ("5e.6 also deletes the legacy `_tts_handler` mirror branch
from `ComposableConversationHandler._clear_queue` setter").

### §3.2 — `_tts_handler` reference audit

Grep across the source tree for `_tts_handler` reads:

```
src/robot_comic/composable_conversation_handler.py:10   docstring reference
src/robot_comic/composable_conversation_handler.py:59   self._tts_handler = tts_handler (constructor)
src/robot_comic/composable_conversation_handler.py:100  docstring reference (5e.6 removes this branch)
src/robot_comic/composable_conversation_handler.py:113  the `_tts_handler` mirror write (5e.6 removes)
src/robot_comic/composable_conversation_handler.py:114  the `_tts_handler` mirror write (5e.6 removes)
src/robot_comic/composable_conversation_handler.py:211  docstring reference
src/robot_comic/composable_conversation_handler.py:215  docstring reference
```

No live source-side **reads** outside the now-deleted `_clear_queue`
mirror. Test-side usage is widespread:

```
tests/test_handler_factory.py:156         assert isinstance(result._tts_handler, ChatterboxTTSResponseHandler)
tests/test_handler_factory.py:222         assert isinstance(result._tts_handler, LlamaElevenLabsTTSResponseHandler)
tests/test_handler_factory.py:289         assert isinstance(result._tts_handler, GeminiTTSResponseHandler)
tests/test_handler_factory_gemini_llm.py:68/76/88/89  isinstance assertions
tests/test_handler_factory_llama_llm.py:56/79         isinstance assertions
tests/test_handler_factory_pipeline_mode.py:112/145   isinstance assertions
tests/integration/test_handler_factory_smoke.py:87    "`result._tts_handler` to confirm the right host was composed"
tests/test_composable_conversation_handler.py:381+    `_clear_queue` mirror tests (5e.6 deletes)
tests/test_composable_persona_reset.py:4/73           5c.2 persona-reset coverage
tests/test_history_trim.py:193                        gemini_tts_handler_trims_history naming only
```

The factory test pattern (`isinstance(result._tts_handler, ...)`) is
how callers verify the wrapper composed the right concrete handler
class. Removing the field would require rewriting every factory test
file's assertions — out of scope for this PR and not a clear win
(the wrapper still legitimately holds the handler instance for its
ABC contract; cf. its `__init__` parameter `tts_handler`).

**Decision: keep `self._tts_handler` on the wrapper.** It serves as
the canonical "what handler did the factory pick" probe for tests
and isn't actively wired to any prod read site after §3.1. Update
the surviving docstrings to reflect that the field is held for
test-side / introspection use only, no longer for any runtime
mirror.

### §3.3 — `LocalSTTInputMixin` decision

The mixin still has two production consumers:

```python
src/robot_comic/local_stt_realtime.py:893
class LocalSTTOpenAIRealtimeHandler(LocalSTTInputMixin, OpenaiRealtimeHandler):
    ...

src/robot_comic/local_stt_realtime.py:934
class LocalSTTHuggingFaceRealtimeHandler(LocalSTTInputMixin, HuggingFaceRealtimeHandler):
    ...
```

These are the realtime-output hybrids — the bundled-realtime LLM+TTS
half lives inside the OpenAI/HF websocket and doesn't decompose into
the STT/LLM/TTS Protocol triple. Per the Phase 4c-tris Option B
"legacy forever" decision (see
`docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md`),
they stay as `LocalSTTInputMixin` subclasses indefinitely.

Two options for the mixin:

- **Option A. Keep the mixin in place** in
  `local_stt_realtime.py:234-890`. The two hybrids inherit from it
  there already; moving it would touch the import line in
  `handler_factory.py` (well, no — `handler_factory.py:71` still
  imports `LocalSTTInputMixin` solely for the `_LocalSTTGeminiTTSHost`
  class that this PR deletes) but is otherwise zero-net-change.
  Update the class docstring to note hybrid-only status.

- **Option B. Move the mixin into the hybrid file.** The mixin
  already lives in `local_stt_realtime.py` — the file where the two
  `LocalSTT*RealtimeHandler` hybrids live. So "move into the hybrids'
  file" is a no-op.

**Decision: Option A.** The mixin stays exactly where it is — same
file as the hybrid handlers — with a docstring update noting it's
hybrid-only post-5e.6. Two source-side changes:

1. The `handler_factory.py:71` import of `LocalSTTInputMixin`
   becomes unused after deleting `_LocalSTTGeminiTTSHost`. Drop the
   import line.
2. The class docstring on `LocalSTTInputMixin` (line 234-235) gets a
   one-line addendum noting it's hybrid-only post-5e.

### §3.4 — `PIPELINE_REFACTOR.md` update

Mark 5e ✅ Done in the Phase 5 status table; add a brief 5e shipping
summary row. Pattern matches the closed 4a-4f rows.

## §4 — Tests

### Part A — gemini_tts triple migration

New / updated assertions in `tests/test_handler_factory.py`:

- `test_moonshine_gemini_tts_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_gemini_tts_wires_should_drop_frame_callback` —
  `result.pipeline.stt._should_drop_frame is not None` and returns
  `False` (handler has no `_speaking_until`; getattr default 0.0).
- `test_moonshine_gemini_tts_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_moonshine_gemini_tts_routes_to_composable`
(`test_handler_factory.py:277`) keeps passing — it asserts
`isinstance(result._tts_handler, GeminiTTSResponseHandler)`; the
plain handler is what the wrapper now wraps.

New leaf-handler idempotency test in `tests/test_gemini_tts_handler.py`
(mirrors 5e.3-5e.5):

- `test_prepare_startup_credentials_is_idempotent` — patch
  `google.genai.Client`; call `_prepare_startup_credentials` twice;
  assert the client is constructed exactly once and the
  `handler._client` instance survives across calls.

### Part B — cleanup

`tests/test_composable_conversation_handler.py`:

- Delete `test_clear_queue_assignment_propagates_to_tts_handler`
  and the matching `_tts_handler is None` assertion in
  `test_clear_queue_assignment_handles_none`.
  The two surviving tests
  (`test_clear_queue_assignment_also_mirrors_onto_pipeline`,
  `test_clear_queue_assignment_mirrors_none_onto_pipeline`) cover
  the live behaviour.
- A new tiny test
  `test_clear_queue_assignment_does_not_touch_tts_handler` asserts
  the wrapper no longer writes onto the host: after the assignment,
  `wrapper._tts_handler._clear_queue` is unchanged (or never set).

## §5 — Verification

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/gemini_tts.py \
    src/robot_comic/handler_factory.py \
    src/robot_comic/composable_conversation_handler.py \
    src/robot_comic/local_stt_realtime.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Known flakes (do NOT fix; re-run if hit):

- `test_huggingface_realtime::test_run_realtime_session_passes_allocated_session_query`
- `test_openai_realtime::test_openai_excludes_head_tracking_when_no_head_tracker`
- `test_handler_factory::test_moonshine_openai_realtime_output`,
  `test_moonshine_hf_output`
- `test_gemini_turn_buffers_transcripts_and_schedules_motion_reset`

## §6 — Risk

Low for Part A (pattern proven across four prior PRs). The triple's
`_speaking_until` no-op echo-guard is documented up-front so no
surprise on the next hardware test.

Low-medium for Part B. Removing the `_tts_handler` mirror branch is
mechanical, but a single un-migrated triple would have been broken by
this change. By construction every composable triple is migrated now
(verified by grep — no `_LocalSTT*Host` classes survive after
deleting `_LocalSTTGeminiTTSHost`). The two `LocalSTT*RealtimeHandler`
hybrids never used the wrapper's `_clear_queue` mirror — they own
their own queue management inside the realtime websocket session.

## §7 — Diff budget

Target: ≤500 LOC across all touched files (well under the brief's
700 LOC trip-wire).

- `gemini_tts.py`: +5 LOC (guard at top + flag at bottom).
- `handler_factory.py`: ~net parity (delete `_LocalSTTGeminiTTSHost`
  + drop the now-unused `LocalSTTInputMixin` import + rewrite
  `_build_composable_gemini_tts`).
- `composable_conversation_handler.py`: -10 LOC net (drop the
  `_tts_handler` mirror branch in the setter + update docstrings).
- `local_stt_realtime.py`: +5 LOC (class docstring addendum noting
  hybrid-only).
- `tests/test_gemini_tts_handler.py`: +35 LOC (one idempotency
  test).
- `tests/test_handler_factory.py`: +50 LOC (three new tests).
- `tests/test_composable_conversation_handler.py`: -15 LOC net
  (drop two tests, add one new tiny assertion).
- `PIPELINE_REFACTOR.md`: small status-table update.

Total: ~250 LOC. Well under budget; no split needed unless surprises
land.
