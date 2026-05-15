# Phase 4c.2 — GeminiLLMAdapter + factory routing for `(moonshine, chatterbox, gemini)`

**Date:** 2026-05-15
**Scope:** New `GeminiLLMAdapter` (~120 LOC); new `_build_composable_gemini_chatterbox` helper in `handler_factory.py`; one new branch in the existing `(moonshine, chatterbox, gemini)` block; tests under `tests/`.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 4c.1 (#361 / `aa59ea1`) — `ChatterboxTTSAdapter` landed; `(moonshine, llama, chatterbox)` routes through `ComposableConversationHandler` under `FACTORY_PATH=composable`.
**Successors:** Phase 4c.3 — `(moonshine, elevenlabs, gemini)`; 4c.4–5 cover the remaining triples.

## Background

`FACTORY_PATH=composable` now covers two triples: `(moonshine, llama, elevenlabs)` and `(moonshine, llama, chatterbox)`. The next-cheapest triple is `(moonshine, chatterbox, gemini)` — the LLM half differs (Gemini API instead of llama-server) but the TTS half (Chatterbox) and STT half (Moonshine) already have adapters.

The legacy handler `GeminiTextChatterboxHandler` (in `gemini_text_handlers.py`) is composed of:

- `LocalSTTInputMixin` — moonshine listener (already adapted by `MoonshineSTTAdapter`).
- `GeminiTextChatterboxResponseHandler(GeminiTextResponseHandler, ChatterboxTTSResponseHandler)` — Gemini-API LLM + chatterbox-server TTS via a diamond MRO.

The LLM half lives in `GeminiTextResponseHandler._call_llm` (in `gemini_text_base.py`). Because `GeminiTextResponseHandler` inherits from `BaseLlamaResponseHandler`, its `_call_llm` returns the *exact same shape* as the llama-server `_call_llm`:

```python
(text: str, tool_calls: list[dict], raw_msg: dict)
```

where each entry in `tool_calls` follows the llama-server convention:

```python
{"index": int, "id": str, "type": "function", "function": {"name": str, "arguments": dict}}
```

The Gemini→llama-server shape conversion happens inside `gemini_llm.py::GeminiLLMClient.call_completion` — it walks the streamed deltas, collects them per-index, and emits llama-server-shaped dicts (including `json.loads(args_str)` to materialise `arguments` as `dict`, lines 322–336). By the time `GeminiTextResponseHandler._call_llm` finishes, the tool-call dicts are already in the canonical shape.

This means `GeminiLLMAdapter` is *structurally identical* to `LlamaLLMAdapter` — both wrap a `BaseLlamaResponseHandler`-derived handler whose `_call_llm` signature and return shape match. The adapter's history-swap, tool-call conversion, and shutdown paths are reusable verbatim. The only meaningful differences are:

1. The type annotation (`GeminiTextResponseHandler` vs `BaseLlamaResponseHandler`).
2. The docstring (Gemini-specific framing for future readers).
3. The credential-prep semantics: `GeminiTextResponseHandler._prepare_startup_credentials` initialises a `GeminiLLMClient` instead of probing `llama-server`. For the chatterbox-MRO subclass, both the Chatterbox HTTP client and the Gemini client are wired in one chain — handled by the legacy handler, the adapter just delegates.

## Goal

Add `GeminiLLMAdapter` and route `(moonshine, chatterbox, gemini)` through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`. Default `legacy` keeps `GeminiTextChatterboxHandler` — zero behaviour change for unset / unchanged operators.

## Out of scope (deferred)

| Item | Sub-phase |
|------|-----------|
| `(moonshine, elevenlabs, gemini)` — `GeminiTextElevenLabsHandler` | 4c.3 |
| `(moonshine, elevenlabs, gemini-fallback)` — `LocalSTTGeminiElevenLabsHandler` | 4c.4 |
| `(moonshine, gemini_tts)` — `LocalSTTGeminiTTSHandler` | 4c.5 (needs `GeminiTTSAdapter`) |
| Sibling `HybridRealtimePipeline` for `LocalSTT*RealtimeHandler` | 4c-tris |
| Flip default to `composable` | 4d |
| Delete legacy handlers | 4e |
| Retire `BACKEND_PROVIDER` | 4f |
| Lifecycle hooks: telemetry, boot-timeline events (#321), joke history, history trim, echo-guard timestamps | Per-hook follow-up PRs |
| First-audio marker / tag plumbing through `LLMResponse.metadata` | Tracked in 4b spec as a known TODO; carries over |

## Design

### 1. `GeminiLLMAdapter`

New file `src/robot_comic/adapters/gemini_llm_adapter.py`. Mirrors `LlamaLLMAdapter` exactly with type annotations updated:

```python
class GeminiLLMAdapter:
    """Adapter exposing ``GeminiTextResponseHandler`` as ``LLMBackend``."""

    def __init__(self, handler: "GeminiTextResponseHandler") -> None:
        self._handler = handler

    async def prepare(self) -> None:
        await self._handler._prepare_startup_credentials()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if tools is not None:
            logger.debug(
                "GeminiLLMAdapter.chat: ignoring %d tools arg (legacy handler "
                "reads tools from deps); see module docstring",
                len(tools),
            )

        saved_history = self._handler._conversation_history
        self._handler._conversation_history = []
        try:
            text, raw_tool_calls, _raw_msg = await self._handler._call_llm(
                extra_messages=messages,
            )
        finally:
            self._handler._conversation_history = saved_history

        tool_calls = tuple(_convert_tool_call(tc) for tc in raw_tool_calls)
        return LLMResponse(text=text, tool_calls=tool_calls)

    async def shutdown(self) -> None:
        http = getattr(self._handler, "_http", None)
        if http is not None:
            try:
                await http.aclose()
            except Exception as exc:
                logger.warning("GeminiLLMAdapter shutdown: aclose() raised: %s", exc)
            self._handler._http = None


def _convert_tool_call(raw: dict[str, Any]) -> ToolCall:
    """Convert a tool_call dict (llama-server shape) to the Protocol's ToolCall.

    GeminiLLMClient.call_completion emits dicts in the llama-server shape
    (id / function.name / function.arguments=dict). This converter is therefore
    identical to LlamaLLMAdapter's; keeping a per-adapter copy avoids the
    cross-module coupling that ``from llama_llm_adapter import _convert_tool_call``
    would introduce.
    """
    fn = raw.get("function", {})
    return ToolCall(
        id=str(raw.get("id", "")),
        name=str(fn.get("name", "")),
        args=fn.get("arguments") or {},
    )
```

Key invariants:

- **Same legacy handler instance is wrapped by all three adapters** (`MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ChatterboxTTSAdapter`), exactly like 4c.1's factory does.
- **History swap** preserves the legacy handler's `_conversation_history` across the adapter call — the orchestrator owns canonical history per the Phase-1 Protocol contract.
- **`tools` arg is accepted for Protocol compliance and dropped.** `GeminiTextResponseHandler._build_llm_messages` reads tool specs from `get_active_tool_specs(self.deps)`, so the adapter has nothing to forward. Same pattern as `LlamaLLMAdapter`.
- **Tool-call conversion is identical to llama's.** Because `gemini_llm.py` already converts Gemini-native `function_call` parts into llama-server-shaped dicts, the `_convert_tool_call` helper is the same as `LlamaLLMAdapter`'s — both extract `id`, `function.name`, `function.arguments`. The conversion lives at the adapter boundary as required.

### 2. `_build_composable_gemini_chatterbox` in `handler_factory.py`

Mirror the existing `_build_composable_llama_chatterbox` helper. The factory branch lives inside the existing `LLM_BACKEND=gemini` + `AUDIO_OUTPUT_CHATTERBOX` arm (currently at `handler_factory.py:230–239`). The composable check must happen *before* the legacy `return GeminiTextChatterboxHandler(...)`.

The new factory branch:

```python
if _llm_backend == LLM_BACKEND_GEMINI:
    if output_backend == AUDIO_OUTPUT_CHATTERBOX:
        if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
            logger.info(
                "HandlerFactory: selecting ComposableConversationHandler "
                "(%s → %s, llm=%s, factory_path=composable)",
                input_backend, output_backend, LLM_BACKEND_GEMINI,
            )
            return _build_composable_gemini_chatterbox(**handler_kwargs)
        from robot_comic.gemini_text_handlers import GeminiTextChatterboxHandler

        logger.info(...)
        return GeminiTextChatterboxHandler(**handler_kwargs)
```

The helper:

```python
def _build_composable_gemini_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, chatterbox, gemini) pipeline."""
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ChatterboxTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = GeminiTextChatterboxHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = GeminiLLMAdapter(legacy)
        tts = ChatterboxTTSAdapter(legacy)
        pipeline = ComposablePipeline(
            stt, llm, tts,
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=legacy,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()
```

### 3. Resolution of open design questions from the briefing

| Question | Resolution |
|----------|-----------|
| Gemini tool-call shape conversion — where does it live? | **Inside `GeminiLLMAdapter._convert_tool_call`** at the adapter boundary, but the converter is structurally identical to `LlamaLLMAdapter._convert_tool_call` because `gemini_llm.py::GeminiLLMClient.call_completion` already converts Gemini's native `function_call` parts into llama-server-shaped dicts (see `gemini_llm.py:283–344`). The adapter never sees Gemini-native shapes. |
| System prompt seeding — risk of double-seed? | `GeminiTextResponseHandler._call_llm` reads `system_instruction` from `get_session_instructions()` (via `_build_llm_messages`) and passes it as `system_instruction` to `GeminiLLMClient.call_completion`. The orchestrator-level `ComposablePipeline.__init__(..., system_prompt=...)` also seeds the system prompt as the first entry in `_conversation_history`. With the history-swap pattern, the handler's `_conversation_history` is empty during the call, so the only path the system prompt reaches the Gemini API is via the legacy `_build_llm_messages` call. **No double-seed.** The orchestrator's seeded system prompt is in the `messages` argument to `chat()` (because `ComposablePipeline` puts it in `_conversation_history` and prepends to the outgoing messages list), but the adapter swaps that as `extra_messages`. `_build_llm_messages` then concatenates an empty `self._conversation_history` with `extra_messages` (which contains the orchestrator's system message), and passes the *result* as `messages` while also passing its own `system_prompt = get_session_instructions()` as `system_instruction` to `call_completion`. `_openai_messages_to_gemini` extracts the role="system" message into `_sys_from_msgs` and discards it from the `contents` list (`gemini_llm.py:57–61`). Then `call_completion` uses `effective_system = system_instruction if system_instruction is not None else _sys_from_msgs` (line 175) — preferring `system_instruction` (which is the legacy-handler-seeded one). The net result: the system prompt the API sees is exactly the one the legacy handler always used. Operator-visible behaviour is unchanged. |
| `prepare()` semantics — does `GeminiTextResponseHandler` expose `_prepare_startup_credentials`? | Yes (`gemini_text_base.py:63–72`). It chains via `super()` so for the diamond subclass `GeminiTextChatterboxResponseHandler` it walks the MRO into `ChatterboxTTSResponseHandler._prepare_startup_credentials` (HTTP client + TTS warmup) and then initialises the Gemini client. The adapter calls it directly — identical to `LlamaLLMAdapter.prepare`. |
| History swap pattern — does `_call_llm` accept `extra_messages`? | Yes (`gemini_text_base.py:125–150`). Signature matches `BaseLlamaResponseHandler._call_llm`. No workaround needed. |
| `tools` arg — adapter ignore? | Yes. `GeminiTextResponseHandler._build_llm_messages` reads `get_active_tool_specs(self.deps)`. Same pattern as `LlamaLLMAdapter`. |

### 4. Why a separate adapter file (not "share `LlamaLLMAdapter`")

The handler types are different: `BaseLlamaResponseHandler` vs `GeminiTextResponseHandler`. We *could* parametrise a single adapter to accept either type, but:

- The `_prepare_startup_credentials` semantics differ (llama probes llama-server health; gemini inits `GeminiLLMClient`). A future change to either won't necessarily apply to the other.
- The type annotations are clearer with one adapter per LLM backend.
- The duplication is ~50 lines and entirely mechanical — far less than the cost of a shared abstraction that has to grow generic enough to satisfy both.
- This matches the pattern set by `LlamaLLMAdapter` / `ElevenLabsTTSAdapter` / `ChatterboxTTSAdapter`: one adapter per legacy class.

A `_convert_tool_call` helper duplicate is the price; tracked in module docstring with a note that 4e (legacy deletion) is the right time to consolidate if desired.

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/adapters/gemini_llm_adapter.py` | NEW — ~120 LOC, mirrors `LlamaLLMAdapter`. |
| `src/robot_comic/adapters/__init__.py` | EDIT — export `GeminiLLMAdapter`. |
| `src/robot_comic/handler_factory.py` | EDIT — composable gate inside the `LLM_BACKEND_GEMINI` + `AUDIO_OUTPUT_CHATTERBOX` arm + new `_build_composable_gemini_chatterbox` helper. ~30 LOC delta. |
| `tests/adapters/test_gemini_llm_adapter.py` | NEW — mirrors `test_llama_llm_adapter.py`: stub handler + prepare / chat / tool-call conversion / history-swap / shutdown / protocol tests. |
| `tests/test_handler_factory_factory_path.py` | EDIT — add legacy-path regression guard + four composable-path tests for `(moonshine, chatterbox, gemini)`. Update the existing `test_composable_path_with_gemini_llm_unchanged` to cover only the *unmigrated* gemini triple (elevenlabs) rather than implying gemini-chatterbox is also unchanged. |

No changes to `composable_pipeline.py`, `composable_conversation_handler.py`, the ABC, or `main.py`.

## Success Criteria

- `FACTORY_PATH=legacy` (default) + `(moonshine, chatterbox, gemini)` → `GeminiTextChatterboxHandler` (bit-for-bit current behaviour).
- `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` whose pipeline holds `MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ChatterboxTTSAdapter` — all wrapping a single `GeminiTextChatterboxHandler` instance.
- `wrapper.copy()` returns a different wrapper whose `_tts_handler` is a different `GeminiTextChatterboxHandler` instance.
- `(moonshine, llama, elevenlabs)` (4b) and `(moonshine, llama, chatterbox)` (4c.1) still return their composable wrappers.
- `(moonshine, elevenlabs, gemini)` and every other unmigrated triple still returns its legacy concrete class.
- Bundled-realtime modes ignore `FACTORY_PATH`.
- `ruff check` / `ruff format --check` / `mypy --pretty` / `pytest tests/ -q` all green from the repo root.

## Test Plan

### Unit tests for `GeminiLLMAdapter` (`tests/adapters/test_gemini_llm_adapter.py`)

Mirrors `test_llama_llm_adapter.py` task-for-task. Same stub-handler approach (records `_call_llm` args, returns scripted tuples):

| Test | Asserts |
|------|---------|
| `test_prepare_calls_handler_prepare` | `adapter.prepare()` calls `_prepare_startup_credentials`. |
| `test_chat_returns_llm_response_with_text` | Plain text response → `LLMResponse(text=..., tool_calls=())`. |
| `test_chat_forwards_messages_as_extra_messages` | `messages` arg arrives at `_call_llm(extra_messages=...)`. |
| `test_chat_clears_handler_history_for_the_call` | Handler's `_conversation_history` is empty inside the call; original is restored after. |
| `test_chat_restores_history_even_when_call_raises` | Exception from `_call_llm` → history still restored. |
| `test_chat_converts_raw_tool_calls_to_protocol_tool_calls` | A llama-server-shaped tool_call dict (which is what `gemini_llm.py` emits) becomes a `ToolCall(id, name, args)`. |
| `test_chat_handles_missing_tool_call_id` | Missing `id` → empty string, no crash. |
| `test_chat_handles_null_tool_call_arguments` | `function.arguments=None` → `args={}`. |
| `test_chat_accepts_and_ignores_tools_kwarg` | `tools=[...]` arg accepted and ignored. |
| `test_shutdown_closes_handler_http` | `_http.aclose()` called and `_http` nulled. |
| `test_shutdown_with_no_open_http_is_safe` | No-op when `_http` is `None`. |
| `test_adapter_satisfies_llm_backend_protocol` | `isinstance(adapter, LLMBackend)` is `True`. |

### Factory dispatch tests (additions to `tests/test_handler_factory_factory_path.py`)

| Test | Asserts |
|------|---------|
| `test_legacy_path_returns_legacy_handler_for_gemini_chatterbox` | `FACTORY_PATH=legacy` + (gemini, chatterbox) → `GeminiTextChatterboxHandler`. |
| `test_composable_path_returns_wrapper_for_gemini_chatterbox` | `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` with `GeminiTextChatterboxHandler` as `_tts_handler`. |
| `test_composable_path_wires_three_adapters_for_gemini_chatterbox` | `pipeline.stt/llm/tts` are `MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ChatterboxTTSAdapter`; all wrap the same legacy instance. |
| `test_composable_path_seeds_system_prompt_for_gemini_chatterbox` | `pipeline._conversation_history[0]` is the patched `get_session_instructions` value. |
| `test_composable_path_copy_constructs_fresh_legacy_for_gemini_chatterbox` | `copy()` produces a new wrapper + a new `GeminiTextChatterboxHandler`. |

Update the existing `test_composable_path_with_gemini_llm_unchanged` test: it currently patches `GeminiTextElevenLabsHandler` for the elevenlabs output and asserts the legacy class is returned. That test still holds (the elevenlabs+gemini triple is 4c.3's responsibility, not 4c.2). The test name already implies "the *unmigrated* gemini triples stay on legacy" — no rename needed.

### What we don't add tests for

- Hardware barge-in (operator validates on robot after merge).
- End-to-end transcript → audio path with real Gemini/Chatterbox servers (covered by the legacy handler's existing test suite; this PR doesn't change those flows).
- Voice switching propagation — already covered by Phase 4a wrapper tests; the chatterbox handler exposes the same surface so existing tests still pass.

## Migration Notes

- Default behaviour is unchanged — operators with `REACHY_MINI_FACTORY_PATH` unset (or `legacy`) keep `GeminiTextChatterboxHandler`.
- Operators who set `FACTORY_PATH=composable` for the chatterbox-gemini triple (zero today on main) will see `ComposableConversationHandler`. Opt-in.
- Reverting is one env-var flip.
- 4c.3 onwards: same shape — reuse `GeminiLLMAdapter` for the elevenlabs+gemini triple, then build `GeminiTTSAdapter` for the gemini-bundled triple in 4c.5.

## Risks

- **Duplicate `_convert_tool_call` helper.** `GeminiLLMAdapter._convert_tool_call` is byte-identical to `LlamaLLMAdapter._convert_tool_call`. Acceptable — both will be consolidated when the legacy adapters are deleted in 4e, or earlier if a future PR introduces a shared `tool_call_conversion` module. Tracked in the module docstring.
- **System-prompt double-seed concern.** Examined in detail under Open Question 2 above. Not a real risk — Gemini's `_openai_messages_to_gemini` extracts and discards the system-role message from `messages`, and the legacy handler's `system_instruction` wins over any inferred one. Pinned by `test_composable_path_seeds_system_prompt_for_gemini_chatterbox` at the orchestrator level (the pipeline's `_conversation_history[0]` is asserted, which is what the orchestrator passes downstream).
- **Diamond MRO subtleties.** `GeminiTextChatterboxResponseHandler` resolves `_prepare_startup_credentials` by overriding it explicitly to call `ChatterboxTTSResponseHandler._prepare_startup_credentials` first, then initialise Gemini. The adapter calls the overridden method on the concrete instance, so the diamond resolves correctly. No special handling needed in the adapter.

## After-merge follow-ups (out of scope for 4c.2)

- 4c.3: reuse `GeminiLLMAdapter` for `(moonshine, elevenlabs, gemini)` — broaden the adapter annotation to accept either Chatterbox or ElevenLabs mixins via `GeminiTextResponseHandler` base.
- 4c.4: `(moonshine, elevenlabs, gemini-fallback)` — `LocalSTTGeminiElevenLabsHandler`. Same shape.
- 4c.5: build `GeminiTTSAdapter` for the gemini-bundled triple.
- Lifecycle hooks: per-PR rollout per the operating manual.
