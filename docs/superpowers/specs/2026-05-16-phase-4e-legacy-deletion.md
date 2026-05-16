# Phase 4e — Delete legacy concrete handlers + retire `FACTORY_PATH`

**Date:** 2026-05-16
**Issue:** [#337](https://github.com/TheKentuckian/robot_comic/issues/337) — Pipeline refactor epic
**Predecessors:** 4d (default flipped to composable, PR #378)
**Operator authorization:** Hardware-validation soak between 4d and 4e LIFTED. Merge on green CI.

## Goal

Delete every `LocalSTT*Handler` concrete class that the composable factory
previously routed through, retire the `FACTORY_PATH` dial entirely, and
simplify `handler_factory.py` so the composable path is the only path.

This is the **largest cleanup PR of the epic** — the wrap-up of the
incremental adapter-and-replace approach (Option C in the Phase 4
exploration memo).

## What we delete

### Concrete handler classes — composable host wrappers

These classes were the `LocalSTTInputMixin + *ResponseHandler` combinations
that the factory previously instantiated to host the three Phase 3 adapters
(`MoonshineSTTAdapter`, an LLM adapter, a TTS adapter):

| File | Class | Why now-removable |
| --- | --- | --- |
| `src/robot_comic/llama_elevenlabs_tts.py` | `LocalSTTLlamaElevenLabsHandler` | Factory composes the mixin inline. |
| `src/robot_comic/chatterbox_tts.py` | `LocalSTTChatterboxHandler` | Same. |
| `src/robot_comic/elevenlabs_tts.py` | `LocalSTTGeminiElevenLabsHandler` + alias `LocalSTTElevenLabsHandler` | Same. |
| `src/robot_comic/gemini_tts.py` | `LocalSTTGeminiTTSHandler` | Same. |
| `src/robot_comic/gemini_text_handlers.py` | `GeminiTextChatterboxHandler`, `GeminiTextElevenLabsHandler` | Same. |
| `src/robot_comic/llama_gemini_tts.py` | `LocalSTTLlamaGeminiTTSHandler` (orphan) | Never reachable from the factory; the orphan-handler bug was fixed in 4c.1 with explicit `LLM_BACKEND=llama` routing. |

### `FACTORY_PATH` dial

- `src/robot_comic/config.py` — `FACTORY_PATH_ENV`, `FACTORY_PATH_LEGACY`,
  `FACTORY_PATH_COMPOSABLE`, `FACTORY_PATH_CHOICES`, `DEFAULT_FACTORY_PATH`,
  `_normalize_factory_path()`, the `FACTORY_PATH` instance attr, and the
  refresh hook in `refresh_runtime_config_from_env`.
- `src/robot_comic/handler_factory.py` — every `if config.FACTORY_PATH ==
  FACTORY_PATH_COMPOSABLE:` branch. Composable becomes the only path.
- `.env.example` — `REACHY_MINI_FACTORY_PATH` block.
- `tests/test_config_factory_path.py` — deleted (the dial is gone).
- `tests/test_handler_factory_factory_path.py` — deleted (the dial is gone).

## What we keep

The adapters delegate into the legacy class hierarchy's internals — those
internals survive untouched:

| Class | Why it survives |
| --- | --- |
| `BaseLlamaResponseHandler` (all internals) | `LlamaLLMAdapter`, `GeminiLLMAdapter`, and the diamond response handlers all delegate to its `_call_llm`, `_prepare_startup_credentials`, `_enqueue_audio_frame`, `_run_turn`, `_dispatch_completed_transcript`, history & tool plumbing. |
| `LlamaElevenLabsTTSResponseHandler` | Hosts the llama-elevenlabs composable triple. |
| `ChatterboxTTSResponseHandler` | `ChatterboxTTSAdapter` delegates `_synthesize_and_enqueue`. |
| `ElevenLabsTTSResponseHandler` | `ElevenLabsTTSAdapter` delegates `_stream_tts_to_queue`. |
| `GeminiTTSResponseHandler` | `GeminiTTSAdapter` + `GeminiBundledLLMAdapter` delegate `_call_tts_with_retry` / `_run_llm_with_tools`. |
| `GeminiTextResponseHandler` | `GeminiLLMAdapter` delegates `_call_llm`. |
| `GeminiTextChatterboxResponseHandler`, `GeminiTextElevenLabsResponseHandler` | The diamond classes that combine the Gemini LLM half with a Chatterbox / ElevenLabs TTS half on one instance — required by `_build_composable_gemini_*` (each instance must satisfy the LLM, TTS, and STT adapters simultaneously). |
| `LlamaGeminiTTSResponseHandler` | Surviving response-handler base for the legacy `LocalSTTLlamaGeminiTTSHandler` orphan. Not factory-reachable but still well-tested (`tests/test_llama_gemini_tts.py`). Kept until a future cleanup decides whether to retire the LLama+Gemini-TTS combo entirely. |
| `LocalSTTInputMixin` | `MoonshineSTTAdapter` depends on the mixin's listener; composed inline by the factory builders into a private subclass per triple. |

All composable code (`ComposablePipeline`, `ComposableConversationHandler`,
`adapters/*.py`) is unchanged in behaviour — only the **handler class** the
factory builders compose changes.

The bundled-realtime path (`HuggingFaceRealtimeHandler`,
`OpenaiRealtimeHandler`, `GeminiLiveHandler`) and the 4c-tris hybrid
handlers (`LocalSTTOpenAIRealtimeHandler`,
`LocalSTTHuggingFaceRealtimeHandler`) are explicitly **not** in scope — they
were Skipped per the operator's Option B decision.

## Factory simplification shape

Each `_build_composable_*` helper today reads as:

```python
from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler
legacy = LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
stt = MoonshineSTTAdapter(legacy)
...
```

After 4e, each helper composes the `LocalSTTInputMixin` on top of the
response-handler base at construction time using a tiny per-triple private
class:

```python
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler

class _LocalSTTLlamaElevenLabsHandler(LocalSTTInputMixin, LlamaElevenLabsTTSResponseHandler):
    """Factory-private mixin host; never exposed."""

legacy = _LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
stt = MoonshineSTTAdapter(legacy)
...
```

The `_dispatch_completed_transcript` MRO-shim override **is** re-added
on each host class because non-adapter call sites (the smoke tests, the
`_send_startup_trigger` "llm" mode) reach the method before the adapter
monkey-patch lands. The shim cost is one async stub per host. Without
it, MRO picks `LocalSTTInputMixin._dispatch_completed_transcript` (an
OpenAI-realtime path that expects `self.connection`) and raises.

The five private host classes are defined at module scope (above the
builder functions) so they cooperate with `type.__init__` only once at
import.

## `_dispatch_completed_transcript` audit

Confirmed via inspection of `MoonshineSTTAdapter.start()`
(`src/robot_comic/adapters/moonshine_stt_adapter.py:55-73`): the adapter
saves the original `_dispatch_completed_transcript` reference, then
overwrites it with a bridge callback **before** invoking
`_prepare_startup_credentials`. The bridge calls the orchestrator's
transcript callback (which feeds `ComposablePipeline._on_transcript_completed`).
The MRO-shim override that the deleted `LocalSTT*Handler` classes carried
(routing past the `LocalSTTInputMixin._dispatch_completed_transcript`
OpenAI-realtime default) is therefore dead weight on the composable path
— the original is restored only in `stop()`, by which point the listener
is being torn down.

## Test rewrite strategy

### Delete entirely

- `tests/test_config_factory_path.py` — the dial is gone.
- `tests/test_handler_factory_factory_path.py` — the dial is gone; the
  scenarios it covered (each composable triple is built correctly) are
  re-expressed in the rewritten `test_handler_factory*.py` files.

### Rewrite to assert `ComposableConversationHandler`

- `tests/test_handler_factory.py` — drop the autouse `FACTORY_PATH=legacy`
  pin. Update each `_assert_combo(..., "module", "LegacyClass")` to assert
  `isinstance(result, ComposableConversationHandler)` and that
  `result._tts_handler` is an instance of the legacy *ResponseHandler base
  (proving the right factory-private subclass was composed).
- `tests/test_handler_factory_llama_llm.py` — same shape.
- `tests/test_handler_factory_gemini_llm.py` — same shape.
- `tests/test_handler_factory_pipeline_mode.py` — drop the autouse pin;
  bundled-realtime cases are unchanged (still concrete classes).

### Update imports only (handler is used for its internals)

These tests build a `LocalSTTChatterboxHandler` (or similar) instance to
exercise inherited `BaseLlamaResponseHandler` internals. The class itself
goes away; the tests are rewired to instantiate the surviving
`*ResponseHandler` base directly:

- `tests/test_llama_base.py` → `ChatterboxTTSResponseHandler`
- `tests/test_llama_streaming.py` → `ChatterboxTTSResponseHandler`
- `tests/test_llama_health_check.py` → `ChatterboxTTSResponseHandler`
- `tests/test_llm_warmup.py` → `ChatterboxTTSResponseHandler`
- `tests/test_history_trim.py` → `ChatterboxTTSResponseHandler` (3 sites)
- `tests/test_echo_suppression.py` → `ChatterboxTTSResponseHandler` (2 sites)
- `tests/test_startup_opener.py` → `ChatterboxTTSResponseHandler`
- `tests/test_llama_gemini_tts.py` → `LlamaGeminiTTSResponseHandler`

The handler instances in these tests never depend on the `LocalSTTInputMixin`
half — they exercise the LLM phase, history trim, etc. So dropping the
mixin from the test fixture is a no-op behaviourally.

### Integration smoke tests

`tests/integration/test_*_smoke.py` are skipped in CI by default and run
end-to-end with real backends. They are migrated to import the surviving
response handlers and compose the mixin inline (mirroring the factory
shape), so they keep exercising the same code paths.

- `tests/integration/test_chatterbox_smoke.py`
- `tests/integration/test_elevenlabs_smoke.py`
- `tests/integration/test_gemini_text_smoke.py`
- `tests/integration/test_llama_elevenlabs_smoke.py`
- `tests/integration/test_handler_factory_smoke.py` — the
  expected-class-name strings update to match the
  `ComposableConversationHandler` wrapper.

## Acceptance criteria

- `git grep "LocalSTTLlamaElevenLabsHandler\|LocalSTTChatterboxHandler\|LocalSTTGeminiElevenLabsHandler\|LocalSTTGeminiTTSHandler\|GeminiTextChatterboxHandler\|GeminiTextElevenLabsHandler\|LocalSTTLlamaGeminiTTSHandler\|LocalSTTElevenLabsHandler"` returns nothing in `src/`.
- `git grep "FACTORY_PATH\|REACHY_MINI_FACTORY_PATH"` returns nothing in `src/`.
- Factory's source-of-truth dispatch matrix lives entirely in the
  composable helpers.
- All remaining tests pass.
- `uvx ruff@0.12.0 check` + `uvx ruff@0.12.0 format --check` green from
  repo root.
- `.venv/Scripts/mypy --pretty <changed files>` green.
- `.venv/Scripts/pytest tests/ -q` green.

## Out of scope

- Retiring `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` — Phase 4f.
- Refactoring `GeminiTTSResponseHandler` to expose `_call_llm` so the
  bundled adapter pair collapses to a single `GeminiLLMAdapter` +
  `GeminiTTSAdapter` — a follow-up.
- Consolidating `_convert_tool_call` between `LlamaLLMAdapter` and
  `GeminiLLMAdapter` — a follow-up; intentional duplication today.
- Deleting `LlamaGeminiTTSResponseHandler` (the orphan's surviving base).
  Not factory-reachable but still has dedicated tests; future cleanup may
  retire the whole Llama+Gemini-TTS combo.

## Rollback

The deletions are mechanical; `git revert` of the PR reinstates each
class. The factory-private host subclasses introduced by 4e use the
exact same MRO (`LocalSTTInputMixin, *ResponseHandler`) as the deleted
classes, so behaviour is preserved bit-for-bit on the composable path.
