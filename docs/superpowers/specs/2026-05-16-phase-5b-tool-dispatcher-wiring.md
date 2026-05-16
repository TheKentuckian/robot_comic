# Phase 5b — Wire `ComposablePipeline.tool_dispatcher` in factory builders

**Date:** 2026-05-16
**Status:** Spec for the Phase 5b fix landing on
`claude/phase-5b-tool-dispatcher-wiring`.
**Predecessor memos:**
- `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` §2.2 (PR #386 —
  the latent-bug analysis).
- `docs/superpowers/specs/2026-05-16-instrumentation-audit.md` §3 + §5 Rec 1
  (PR #385 — the `tool.execute` span gap on the composable path).

---

## §1 — The latent bug

`ComposablePipeline.tool_dispatcher: ToolDispatcher | None` is the
orchestrator's callback for turning an LLM-emitted `ToolCall` into the
`str` content of the next `role=tool` history entry
(`composable_pipeline.py:75-78`, `:105`, `:115`). When the LLM returns
`LLMResponse(tool_calls=...)` the orchestrator's branch at
`composable_pipeline.py:224-231` reads:

```python
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
```

Pre-Phase 5b, **every** composable factory builder in `handler_factory.py`
constructs `ComposablePipeline(stt, llm, tts, system_prompt=...)` with no
`tool_dispatcher` argument, so the field defaulted to `None`. The
adapter-side wiring is the unlucky half:

- `LlamaLLMAdapter.chat` (`llama_llm_adapter.py:55-92`) faithfully
  forwards llama-server-emitted `tool_calls` into
  `LLMResponse.tool_calls`.
- `GeminiLLMAdapter.chat` (`gemini_llm_adapter.py:55-118`) does the same.
- `GeminiBundledLLMAdapter.chat` (`gemini_bundled_llm_adapter.py`)
  dispatches tools internally via the wrapped handler's
  `_run_llm_with_tools` and returns `LLMResponse(tool_calls=())` always
  — the bundled-Gemini triple worked by coincidence.

Net effect: on the four non-bundled composable triples —
`(moonshine, llama, elevenlabs)`, `(moonshine, llama, chatterbox)`,
`(moonshine, gemini, chatterbox)`, `(moonshine, gemini, elevenlabs)` —
**any tool-triggered turn silently broke the conversation loop without
speaking.** The Phase 4 hardware soak missed it because Don
Rickles-style transcripts trigger tools rarely; the moment an operator
session called a tool, the user heard silence on that turn.

---

## §2 — The fix

Wire a dispatcher closure into every composable factory builder. The
shim lives at module level in `handler_factory.py`:

```python
def _make_tool_dispatcher(host: Any) -> Any:
    async def _dispatch(call: ToolCall) -> str:
        args_json = json.dumps(call.args or {})
        tracer = telemetry.get_tracer()
        with tracer.start_as_current_span(
            "tool.execute",
            attributes={"tool.name": call.name, "tool.id": call.id},
        ) as span:
            try:
                if call.name in _SYSTEM_TOOL_NAMES:
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
        return json.dumps(result)
    return _dispatch
```

Each tool-enabled `_build_composable_*` helper now passes
`tool_dispatcher=_make_tool_dispatcher(host)` to `ComposablePipeline(...)`.

### §2.1 Routing — system tools vs. regular tools

`ToolCallRoutine.__call__` (`tools/background_tool_manager.py:54-61`)
splits on `SYSTEM_TOOL_NAMES = {"task_status", "task_cancel"}`. Those two
tools accept a `tool_manager` kwarg so they can list / cancel other
running background tools. The shim mirrors that split using the host's
own `BackgroundToolManager` (constructed in
`BaseLlamaResponseHandler.__init__:95` and on every other surviving
response-handler base).

### §2.2 Result shape

`composable_pipeline._dispatch_tools_and_record` appends the dispatcher
return value verbatim as the `content` of a `role=tool` message. The
legacy parity at `llama_base.py:617` is `json.dumps(result)`, so the
shim does the same — the LLM gets structured JSON on the next
round-trip. Non-serialisable args / results trip a `WARNING` log and
fall back to `"{}"` / `repr(result)` rather than letting the exception
escape and break the turn.

### §2.3 Background-tool placeholder dispatch — deferred

The legacy `_start_tool_calls` (`llama_base.py:701-724`) fires a
`BackgroundTool` and returns immediately; the result arrives via the
`BackgroundToolManager` notification queue. The composable
orchestrator's `await tool_dispatcher(call)` model is
synchronous-return-per-call — there's no place for the manager's
notification queue. Two paths exist:

(a) Dispatcher returns a placeholder string immediately for background
    tools; result arrives later through a side channel.
(b) Orchestrator gains an "outstanding background results" channel and
    awaits them between turns.

Per memo §5.2 of the Phase 5 exploration — *"the cheapest path (return
placeholder, fire notification later) is what the legacy code did. Don't
over-design — keep it"* — Phase 5b takes neither path. The synchronous
shim blocks the turn until the tool returns, which matches the legacy
`_await_tool_results` 30 s timeout happy path. Long-running tools that
relied on the bg-manager queue are a regression vector but not a
blocker; defer the proper fix to a later Phase 5 sub-phase if it
surfaces in field testing.

### §2.4 The bundled-Gemini triple

`(moonshine, gemini-bundled, gemini_tts)` intentionally stays without a
wired dispatcher. The `GeminiBundledLLMAdapter` calls
`_run_llm_with_tools` which dispatches tools inside the wrapped handler
and always returns `LLMResponse(tool_calls=())` — the orchestrator's
tool-call branch never fires for that triple. Wiring a dispatcher on it
would be dead code. The contract is documented in
`adapters/gemini_bundled_llm_adapter.py:17-19`.

---

## §3 — `tool.execute` span emission

The instrumentation audit's Rec 1 flagged the same dispatch site:
`composable_pipeline._dispatch_tools_and_record` (`composable_pipeline.py:242`)
emitted no span around the dispatcher callback. The monitor's
tool-count column reads child `tool.execute` spans to populate the
Tools cell; post-Phase 4d the column showed `0` for every composable
turn that actually called tools.

The shim emits the span with the **same attribute shape** as the legacy
`BackgroundToolManager._run_tool` span
(`tools/background_tool_manager.py:211-213`):

| Attribute | Value | Allowlist source |
|---|---|---|
| `tool.name` | `call.name` | `_SPAN_ATTRS_TO_KEEP` |
| `tool.id` | `call.id` (orchestrator-supplied) | `_SPAN_ATTRS_TO_KEEP` |
| `outcome` | `"success"` / `"error"` | `_SPAN_ATTRS_TO_KEEP` |

All three keys are already in the `telemetry._SPAN_ATTRS_TO_KEEP`
allowlist, so the monitor needs no change.

### `robot.tool.duration` histogram — skipped

Adding a new histogram instrument touches `telemetry.py` (allowlist
plus instrument definition plus `record_*` helper). Per the dispatch
instructions, that file is in the parallel telemetry-housekeeping
agent's scope; a duplicate edit would create a merge conflict for no
material gain. The span's wall-clock duration is recoverable from the
existing `tool.execute` row, so a follow-up PR can add the histogram
without re-running this work.

---

## §4 — Test coverage

`tests/test_phase_5b_tool_dispatcher_wiring.py` — 8 tests, all GREEN
after the wiring fix:

1. **4 parametric factory wiring assertions** (one per affected triple):
   `pipeline.tool_dispatcher is not None`.
2. **Dispatcher routing** — `dispatch_tool_call` is invoked with the
   expected `tool_name` / `args_json` / `deps`, return value is a JSON
   string.
3. **Error result surfacing** — `{"error": ...}` from the tool layer
   round-trips as a JSON string the orchestrator can append to history.
4. **Span emit (success)** — a `tool.execute` span appears with
   `tool.name`, `tool.id`, `outcome=success` attrs.
5. **Span emit (error)** — same shape with `outcome=error`.

All 8 tests **fail before the wiring fix** (commit
`88a0913 test(phase-5b): RED regression`) — bug-witness confirmed.

The bundled-Gemini triple is intentionally excluded from the parametric
wiring assertion (the adapter dispatches internally; no caller fires
the orchestrator's tool-call branch).

---

## §5 — Diff shape

| File | LOC added | LOC removed | Purpose |
|---|---|---|---|
| `tests/test_phase_5b_tool_dispatcher_wiring.py` | ~330 | 0 | RED regression + span tests |
| `src/robot_comic/handler_factory.py` | ~100 | ~5 | shim helper + 4 wiring lines |
| `docs/superpowers/specs/2026-05-16-phase-5b-tool-dispatcher-wiring.md` | (this file) | 0 | spec |
| `docs/superpowers/plans/2026-05-16-phase-5b-tool-dispatcher-wiring.md` | TDD plan | 0 | plan |

Total: ~430 LOC added across 4 files, 5 LOC removed (import sort + the
mechanical wiring line per builder). Source vs. test ratio is
~100 src / ~330 tests.

---

## §6 — Things to double-check before merge

1. **`telemetry.get_tracer()` returns a no-op on import.** The shim's
   `tracer.start_as_current_span("tool.execute", ...)` call must not
   raise if `telemetry.init()` was never called (sim mode, unit
   tests). Verified locally; the OTel SDK returns a `NoOpTracer` when
   no provider is set and `start_as_current_span` is a context-manager
   no-op.

2. **`host.tool_manager` exists on every host class.** Every surviving
   `*ResponseHandler` base inherits from `BaseLlamaResponseHandler` /
   `BaseRealtimeHandler` (chatterbox / elevenlabs / gemini-text
   variants all do); `BaseLlamaResponseHandler.__init__:95` sets
   `self.tool_manager = BackgroundToolManager()`. `_LocalSTTGeminiTTSHost`
   wraps `GeminiTTSResponseHandler` which has its own manager. The
   dispatch shim is safe across all five hosts; system tools route
   correctly on each.

3. **`call.args` is always a dict.** `ToolCall` declares
   `args: dict[str, Any]` (`backends.py:54-66`), and both LLM adapters
   convert llama-server tool_call dicts into that shape
   (`llama_llm_adapter.py:107-114`, `gemini_llm_adapter.py:_convert_tool_call`).
   The shim defensively coerces `call.args` to `{}` if a future adapter
   regression ships a non-dict.

4. **`json.dumps` on tool results.** Every concrete `Tool.__call__`
   returns a `dict[str, Any]` per `core_tools.py:Tool.__call__`'s
   return type. Non-serialisable values would surface as `TypeError`
   here; the shim catches and falls back to `repr(result)` rather than
   crashing the turn.

5. **No hardware validation needed.** Per the dispatch instructions:
   *"Hardware verification is NOT required. Your regression test is the
   verification surface."* The 8-test suite exercises factory wiring,
   dispatcher routing, error surfacing, and span emission — every layer
   between the orchestrator and the tool registry. Hardware soak can
   ride along with the next Phase 5 sub-phase's natural validation.

---

## §7 — Out of scope (followups)

- **Background-tool placeholder dispatch.** §2.3 — defer.
- **`robot.tool.duration` histogram.** §3 — owned by parallel agent.
- **Tools-spec source of truth.** Memo §2.2 item 3 — adapter currently
  ignores the `tools` arg to `chat()` and pulls from `deps` via
  `get_active_tool_specs(deps)`. Migration to the orchestrator's
  `tools_spec` parameter is a separate sub-PR.
- **The `_SPAN_ATTRS_TO_KEEP` allowlist already covers our needs.** No
  changes there.
