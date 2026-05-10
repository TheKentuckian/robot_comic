# Design: Feed Tool Results Back to LLM in Chatterbox Handler

**Issue**: #48  
**Date**: 2026-05-10  
**Status**: Approved

## Problem

Tool calls in `ChatterboxTTSResponseHandler` are fire-and-forget. For query tools like `camera`, results (e.g. a visual description of the person) are never fed back to the LLM, so the robot cannot reason about or speak to what those tools returned. Action tools (dance, emotion) need no feedback loop.

## Scope

- Add two-phase response cycle: speak immediately, then follow up once query tool results arrive
- One follow-up LLM pass per turn (no recursive tool loops)
- Action tools stay fire-and-forget — no behavioral change for dance, emotion, move_head
- History format corrected: assistant messages with tool calls must include the `tool_calls` structure for Ollama multi-turn

Out of scope: multi-round tool loops (`_LLM_MAX_TOOL_ROUNDS`), streaming, camera vision image path handling.

## Approach

**Approach A — await all, filter by result content.** All tools fire via `BackgroundToolManager` as today. After speaking phase-1 text, await all tool tasks concurrently with a timeout. Filter to "meaningful" results (any string value in the result dict longer than `_MEANINGFUL_RESULT_MIN_LEN` chars). Feed meaningful results to a second LLM call and speak the follow-up. No hardcoded query-tool registry — filtering is content-driven, so future tools that return data automatically participate.

## Data Flow

### Phase 1 (immediate — unchanged latency)

1. Append `{"role": "user", "content": transcript}` to `_conversation_history`
2. `_call_llm()` → `(text, tool_calls)`. `_call_llm` now also returns the raw Ollama `message` dict.
3. Append the raw assistant message to history (preserves `tool_calls` field for Ollama multi-turn format)
4. `_start_tool_calls(tool_calls)` → `list[tuple[call_id, BackgroundTool]]` (replaces void `_dispatch_tool_calls`)
5. Synthesize and enqueue TTS for `text` immediately — robot starts speaking without waiting

### Phase 2 (query feedback — only when tools fired)

6. `await _await_tool_results(bg_tools)` — `asyncio.gather` with `asyncio.wait_for(..., timeout=_TOOL_RESULT_TIMEOUT)` per task; returns `dict[call_id, result]` for tasks that completed within timeout
7. Filter to meaningful results: `any(isinstance(v, str) and len(v) > _MEANINGFUL_RESULT_MIN_LEN for v in result.values())`
8. If any meaningful results: append `{"role": "tool", "content": json.dumps(result), "tool_call_id": call_id}` for each
9. Second `_call_llm()` → `(follow_up_text, _)` (tool calls on this pass are ignored — tools already ran)
10. Append `{"role": "assistant", "content": follow_up_text}` to history
11. Synthesize and enqueue TTS for `follow_up_text`

If no meaningful results (all action tools, or all timed out): skip steps 8–11 entirely.

## New Constants

```python
_TOOL_RESULT_TIMEOUT: float = 5.0       # seconds to await each tool task
_MEANINGFUL_RESULT_MIN_LEN: int = 20    # min chars in any string value to consider meaningful
```

## API Changes

### `_start_tool_calls(tool_calls) -> list[tuple[str, BackgroundTool]]`

Replaces `_dispatch_tool_calls`. Same dispatch logic; now returns `(call_id, bg_tool)` pairs so callers can await tasks.

### `_await_tool_results(bg_tools, timeout) -> dict[str, dict]`

New method. Awaits each `bg_tool._task` via `asyncio.wait_for`. Timeouts and exceptions are swallowed — a failed/timed-out tool simply contributes no result. Returns `{call_id: result_dict}` for completed tools.

### `_call_llm() -> tuple[str, list[dict], dict]`

Signature change: now returns `(text, tool_calls, raw_message)` where `raw_message` is the `message` dict from Ollama's response. Callers that previously ignored the third value can use `_` for it.

### `_dispatch_completed_transcript`

Restructured to orchestrate the two-phase flow above.

## Error Handling

| Scenario | Behavior |
|---|---|
| Tool task times out | Silently excluded from results; tool continues running in background |
| Tool task raises exception | Same as timeout |
| Phase-2 LLM call fails | Log warning; no second speech turn (phase-1 already delivered) |
| `text` empty and no meaningful results | Silent turn — same as today |
| `text` empty but meaningful results arrive | Phase-2 LLM call still fires; robot speaks only the follow-up |

## History Integrity

Currently: `{"role": "assistant", "content": text}` is appended even when tool calls happened. Ollama's multi-turn format requires the assistant message to include `tool_calls` when they were present. Fix: save the raw Ollama `message` dict directly (it already has the right shape). When no tool calls, the dict is `{"role": "assistant", "content": text}` — identical to current behavior.

## Testing Plan

1. **`test_start_tool_calls_returns_bg_tools`** — verify `_start_tool_calls` returns `(call_id, BackgroundTool)` pairs, one per tool call
2. **`test_await_tool_results_returns_completed`** — mock `BackgroundTool._task` as instant coroutine returning a result; verify `_await_tool_results` returns it
3. **`test_await_tool_results_timeout_excluded`** — mock a task that never completes; verify it is absent from returned dict
4. **`test_meaningful_result_camera_passes`** — result with a long string value passes the filter
5. **`test_meaningful_result_action_filtered`** — empty dict or `{"status": "ok"}` does not pass
6. **`test_second_llm_pass_fires_on_meaningful_result`** — end-to-end mock: camera returns a description, verify two `_http.post` calls are made and two TTS calls are made
7. **`test_no_second_pass_for_action_tools`** — dance returns `{}`; verify only one `_http.post` call
8. **`test_tool_message_appended_to_history`** — verify `{"role": "tool", ...}` appears in the messages sent on the second LLM call
9. **`test_assistant_message_preserves_tool_calls_field`** — verify history entry includes `tool_calls` when Ollama returned them
10. All 40 existing chatterbox tests must continue to pass
