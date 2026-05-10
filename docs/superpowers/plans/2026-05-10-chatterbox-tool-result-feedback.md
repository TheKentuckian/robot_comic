# Two-Phase Tool Result Feedback in ChatterboxTTSResponseHandler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed query tool results (e.g. camera descriptions) back to Ollama in a second LLM pass so the robot can reason about and speak to what those tools returned.

**Architecture:** Two-phase response cycle. Phase 1 speaks immediately from the first LLM call and fires all tools. Phase 2 awaits tool tasks with a timeout, filters to meaningful string results (camera descriptions pass; dance `{}` does not), appends `role: tool` messages to history, and runs a second LLM call + TTS pass when any meaningful results arrived.

**Tech Stack:** Python asyncio, httpx, Ollama `/api/chat`, `BackgroundToolManager` (existing), pytest-asyncio

---

## File Map

- Modify: `src/robot_comic/chatterbox_tts.py` — all implementation changes
- Modify: `tests/test_chatterbox_tts.py` — 9 new tests + update 5 existing mocks for 3-tuple `_call_llm` return

---

### Task 1: Add module constants; change `_call_llm` to return `(text, tool_calls, raw_message)`

`_call_llm` currently returns a 2-tuple. It needs a 3rd element — a `dict` shaped for direct `_conversation_history` append — so that the assistant message always carries `tool_calls` when present (required for Ollama multi-turn).

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py`
- Modify: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_chatterbox_tts.py` (before the `_drain_queue` helper):

```python
@pytest.mark.asyncio
async def test_call_llm_returns_raw_message() -> None:
    """_call_llm returns a 3-tuple; the third element is the raw assistant message dict."""
    import httpx
    handler = _make_handler()

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Hey there!",
            "tool_calls": [],
        }
    }
    handler._http.post = AsyncMock(return_value=fake_resp)

    result = await handler._call_llm()

    assert len(result) == 3
    text, tool_calls, raw_message = result
    assert text == "Hey there!"
    assert raw_message["role"] == "assistant"
    assert "content" in raw_message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_call_llm_returns_raw_message -v`
Expected: FAIL — `ValueError: not enough values to unpack` (currently returns 2-tuple)

- [ ] **Step 3: Add module-level constants**

In `src/robot_comic/chatterbox_tts.py`, add after `_TOOL_USE_ADDENDUM` (around line 65):

```python
_TOOL_RESULT_TIMEOUT: float = 5.0
_MEANINGFUL_RESULT_MIN_LEN: int = 20
```

- [ ] **Step 4: Change `_call_llm` return type and build `raw_msg`**

Replace the `_call_llm` method signature and its return statements. The method body is unchanged except for building `raw_msg` before each `return`:

```python
async def _call_llm(self) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Call Ollama /api/chat with tool specs; returns (text, tool_calls, raw_message)."""
    assert self._http is not None
    system_prompt = get_session_instructions() + _TOOL_USE_ADDENDUM
    tool_specs = get_active_tool_specs(self.deps)
    ollama_tools = [self._trim_tool_spec(s) for s in tool_specs]
    messages = [{"role": "system", "content": system_prompt}] + self._conversation_history
    logger.info(
        "_call_llm: profile=%r model=%s tools=%d sys_chars=%d sys_head=%r",
        getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
        getattr(config, "OLLAMA_MODEL", "hermes3:8b-llama3.1-q4_K_M"),
        len(ollama_tools),
        len(system_prompt),
        system_prompt[:80],
    )

    payload: dict[str, Any] = {
        "model": getattr(config, "OLLAMA_MODEL", "hermes3:8b-llama3.1-q4_K_M"),
        "messages": messages,
        "tools": ollama_tools,
        "stream": False,
    }

    nudge_attempted = False

    delay = _LLM_RETRY_BASE_DELAY
    for attempt in range(_LLM_MAX_RETRIES):
        try:
            r = await self._http.post(
                f"{self._ollama_base_url}/api/chat",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {})
            text = (msg.get("content") or "").strip()
            tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

            text, tool_calls = self._coerce_text_tool_call(text, tool_calls)

            if not tool_calls and not text and not nudge_attempted:
                nudge_attempted = True
                logger.info("Hermes3 returned empty response — attempting nudge")
                text, tool_calls = await self._nudge_llm(messages, payload)

            raw_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                raw_msg["tool_calls"] = tool_calls
            return text, tool_calls, raw_msg
        except Exception as exc:
            if attempt == _LLM_MAX_RETRIES - 1:
                raise
            logger.warning("LLM attempt %d/%d failed: %s: %s; retrying in %.1fs",
                           attempt + 1, _LLM_MAX_RETRIES, type(exc).__name__, exc, delay)
            await asyncio.sleep(delay)
            delay *= 2
    return "", [], {}
```

- [ ] **Step 5: Update `_dispatch_completed_transcript` to unpack 3-tuple**

In `_dispatch_completed_transcript`, change:

```python
response_text, tool_calls = await self._call_llm()
```
to:
```python
response_text, tool_calls, raw_message = await self._call_llm()
```

Change:
```python
self._conversation_history.append({"role": "assistant", "content": response_text})
```
to:
```python
self._conversation_history.append(raw_message)
```

- [ ] **Step 6: Update 5 existing `fake_llm` mocks in tests to return 3-tuples**

In `tests/test_chatterbox_tts.py`, find every `async def fake_llm()` that returns a 2-tuple and add `{}` as the third element. There are exactly 5:

`test_tts_called_once_per_sentence`:
```python
async def fake_llm() -> tuple[str, list, dict]:
    return "Hello! How are you today?", [], {}
```

`test_frames_emitted_per_sentence_not_buffered`:
```python
async def fake_llm() -> tuple[str, list, dict]:
    return "First sentence. Second sentence.", [], {}
```

`test_single_sentence_still_produces_audio`:
```python
async def fake_llm() -> tuple[str, list, dict]:
    return "Hiya!", [], {}
```

`test_tts_error_on_all_sentences_pushes_error_output`:
```python
async def fake_llm() -> tuple[str, list, dict]:
    return "Hello. World.", [], {}
```

`test_split_text_disabled_in_tts_payload`:
```python
async def fake_llm() -> tuple[str, list, dict]:
    return "Hello. World.", [], {}
```

Also update `test_call_llm_detects_json_content_tool_call` (line 434) to unpack 3 values:
```python
text, tool_calls, _ = await handler._call_llm()
```

- [ ] **Step 7: Run all tests to verify they pass**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py -v`
Expected: all tests pass (40+ green, 0 failures)

- [ ] **Step 8: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat(chatterbox): _call_llm returns raw_message; add _TOOL_RESULT_TIMEOUT + _MEANINGFUL_RESULT_MIN_LEN (#48)"
```

---

### Task 2: Rename `_dispatch_tool_calls` → `_start_tool_calls` (returns BackgroundTool pairs)

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py`
- Modify: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_chatterbox_tts.py`:

```python
@pytest.mark.asyncio
async def test_start_tool_calls_returns_bg_tools() -> None:
    """_start_tool_calls returns (call_id, BackgroundTool) pairs, one per tool call."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()

    async def fake_start_tool(call_id, tool_call_routine, is_idle_tool_call):
        bg = BackgroundTool(
            id=call_id,
            tool_name=tool_call_routine.tool_name,
            is_idle_tool_call=False,
            status=ToolState.RUNNING,
        )
        return bg

    handler.tool_manager.start_tool = fake_start_tool  # type: ignore[method-assign]

    tool_calls = [
        {"function": {"name": "dance", "arguments": {"style": "wave"}}},
        {"function": {"name": "play_emotion", "arguments": {"emotion": "happy1"}}},
    ]
    result = await handler._start_tool_calls(tool_calls)

    assert len(result) == 2
    for call_id, bg_tool in result:
        assert isinstance(call_id, str) and len(call_id) == 8
        assert isinstance(bg_tool, BackgroundTool)
    assert {bg.tool_name for _, bg in result} == {"dance", "play_emotion"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_start_tool_calls_returns_bg_tools -v`
Expected: FAIL — `AttributeError: '_start_tool_calls' not found`

- [ ] **Step 3: Replace `_dispatch_tool_calls` with `_start_tool_calls`**

In `src/robot_comic/chatterbox_tts.py`, replace the entire `_dispatch_tool_calls` method with:

```python
async def _start_tool_calls(
    self, tool_calls: list[dict[str, Any]]
) -> "list[tuple[str, BackgroundTool]]":
    """Dispatch tool calls; return (call_id, BackgroundTool) pairs."""
    from robot_comic.tools.background_tool_manager import BackgroundTool
    results: list[tuple[str, BackgroundTool]] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        args = fn.get("arguments", {})
        args_json = json.dumps(args) if isinstance(args, dict) else str(args)
        call_id = uuid.uuid4().hex[:8]
        try:
            bg_tool = await self.tool_manager.start_tool(
                call_id=call_id,
                tool_call_routine=ToolCallRoutine(
                    tool_name=tool_name,
                    args_json_str=args_json,
                    deps=self.deps,
                ),
                is_idle_tool_call=False,
            )
            results.append((call_id, bg_tool))
            logger.info("Dispatched tool: %s (call_id=%s)", tool_name, call_id)
        except Exception as exc:
            logger.warning("Failed to dispatch tool %s: %s", tool_name, exc)
    return results
```

- [ ] **Step 4: Update the call site in `_dispatch_completed_transcript`**

Change:
```python
if tool_calls:
    await self._dispatch_tool_calls(tool_calls)
```
to:
```python
if tool_calls:
    await self._start_tool_calls(tool_calls)
```
(The return value is discarded for now; Task 4 will capture it.)

- [ ] **Step 5: Run all tests to verify they pass**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat(chatterbox): _start_tool_calls returns BackgroundTool pairs (#48)"
```

---

### Task 3: Add `_is_meaningful_result` and `_await_tool_results`

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py`
- Modify: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write failing tests for `_is_meaningful_result`**

Add to `tests/test_chatterbox_tts.py`:

```python
# ---------------------------------------------------------------------------
# _is_meaningful_result
# ---------------------------------------------------------------------------

def test_meaningful_result_camera_passes() -> None:
    """A result with a long string value passes the meaningful filter."""
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    result = {
        "description": "A person is standing in the center of the frame, looking directly at the camera."
    }
    assert ChatterboxTTSResponseHandler._is_meaningful_result(result) is True


def test_meaningful_result_action_filtered() -> None:
    """Empty dict or short-value dict does not pass the meaningful filter."""
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    assert ChatterboxTTSResponseHandler._is_meaningful_result({}) is False
    assert ChatterboxTTSResponseHandler._is_meaningful_result({"status": "ok"}) is False
    assert ChatterboxTTSResponseHandler._is_meaningful_result({"status": "done", "count": 3}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_meaningful_result_camera_passes tests/test_chatterbox_tts.py::test_meaningful_result_action_filtered -v`
Expected: FAIL — `AttributeError: '_is_meaningful_result'`

- [ ] **Step 3: Implement `_is_meaningful_result`**

Add as a static method on `ChatterboxTTSResponseHandler`, after `_is_coerce_text_tool_call` / before `_call_llm`:

```python
@staticmethod
def _is_meaningful_result(result: dict[str, Any]) -> bool:
    return any(
        isinstance(v, str) and len(v) > _MEANINGFUL_RESULT_MIN_LEN
        for v in result.values()
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_meaningful_result_camera_passes tests/test_chatterbox_tts.py::test_meaningful_result_action_filtered -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for `_await_tool_results`**

Add to `tests/test_chatterbox_tts.py`:

```python
# ---------------------------------------------------------------------------
# _await_tool_results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_await_tool_results_returns_completed() -> None:
    """Completed tool tasks have their results returned."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    expected_result = {
        "description": "A smiling person stands before a plain background, facing the camera."
    }

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="abc123",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result=expected_result,
    )
    bg_tool._task = asyncio.create_task(instant_task())

    results = await handler._await_tool_results([("abc123", bg_tool)], timeout=1.0)

    assert results == {"abc123": expected_result}


@pytest.mark.asyncio
async def test_await_tool_results_timeout_excluded() -> None:
    """A tool that doesn't finish within the timeout is excluded from results."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()

    async def never_finishes() -> None:
        await asyncio.sleep(9999)

    bg_tool = BackgroundTool(
        id="slow1",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.RUNNING,
    )
    bg_tool._task = asyncio.create_task(never_finishes())

    results = await handler._await_tool_results([("slow1", bg_tool)], timeout=0.05)

    assert results == {}

    bg_tool._task.cancel()
    try:
        await bg_tool._task
    except (asyncio.CancelledError, Exception):
        pass
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_await_tool_results_returns_completed tests/test_chatterbox_tts.py::test_await_tool_results_timeout_excluded -v`
Expected: FAIL — `AttributeError: '_await_tool_results'`

- [ ] **Step 7: Implement `_await_tool_results`**

Add as an instance method on `ChatterboxTTSResponseHandler`, after `_start_tool_calls`:

```python
async def _await_tool_results(
    self,
    bg_tools: "list[tuple[str, BackgroundTool]]",
    timeout: float = _TOOL_RESULT_TIMEOUT,
) -> dict[str, dict[str, Any]]:
    """Await all tool tasks concurrently; return results that arrived within timeout.

    asyncio.shield prevents task cancellation on timeout — tool continues in background.
    """
    from robot_comic.tools.background_tool_manager import BackgroundTool

    async def _wait_one(
        call_id: str, bg_tool: BackgroundTool
    ) -> tuple[str, dict[str, Any] | None]:
        if bg_tool._task is None:
            return call_id, None
        try:
            await asyncio.wait_for(asyncio.shield(bg_tool._task), timeout=timeout)
            return call_id, bg_tool.result
        except Exception:
            return call_id, None

    pairs = await asyncio.gather(*(_wait_one(cid, bt) for cid, bt in bg_tools))
    return {cid: result for cid, result in pairs if result is not None}
```

- [ ] **Step 8: Run all new tests to verify they pass**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_await_tool_results_returns_completed tests/test_chatterbox_tts.py::test_await_tool_results_timeout_excluded tests/test_chatterbox_tts.py::test_meaningful_result_camera_passes tests/test_chatterbox_tts.py::test_meaningful_result_action_filtered -v`
Expected: PASS

- [ ] **Step 9: Run full suite to catch regressions**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py -v`
Expected: all tests pass

- [ ] **Step 10: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat(chatterbox): add _await_tool_results and _is_meaningful_result (#48)"
```

---

### Task 4: Extract `_synthesize_and_enqueue`; restructure `_dispatch_completed_transcript` for two-phase flow

This is where the feature comes together. Four tests exercise the new behavior end-to-end.

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py`
- Modify: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write four failing tests**

Add to `tests/test_chatterbox_tts.py`:

```python
# ---------------------------------------------------------------------------
# Two-phase tool result feedback (end-to-end)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_llm_pass_fires_on_meaningful_result() -> None:
    """Camera returns a long description → two LLM calls, two TTS calls."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    llm_call_count = 0
    tts_texts: list[str] = []
    camera_result = {
        "description": "A young woman with curly red hair stands close to the camera, grinning wide."
    }

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="cam1",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result=camera_result,
    )
    bg_tool._task = asyncio.create_task(instant_task())

    async def patched_llm():
        nonlocal llm_call_count
        llm_call_count += 1
        if llm_call_count == 1:
            return (
                "Let me take a look!",
                [{"function": {"name": "camera", "arguments": {}}}],
                {"role": "assistant", "content": "Let me take a look!",
                 "tool_calls": [{"function": {"name": "camera", "arguments": {}}}]},
            )
        return "Oh yeah, I see a grinning woman!", [], {"role": "assistant", "content": "Oh yeah, I see a grinning woman!"}

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        tts_texts.append(text)
        return _pcm_bytes(2400)

    async def fake_start_tool_calls(tool_calls):
        return [("cam1", bg_tool)]

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("what do you see?")

    assert llm_call_count == 2, f"Expected 2 LLM calls, got {llm_call_count}"
    assert len(tts_texts) >= 2


@pytest.mark.asyncio
async def test_no_second_pass_for_action_tools() -> None:
    """Dance returns {} → only one LLM call is made."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    llm_call_count = 0

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="dance1",
        tool_name="dance",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={},
    )
    bg_tool._task = asyncio.create_task(instant_task())

    async def patched_llm():
        nonlocal llm_call_count
        llm_call_count += 1
        return (
            "I'll dance for you!",
            [{"function": {"name": "dance", "arguments": {}}}],
            {"role": "assistant", "content": "I'll dance for you!",
             "tool_calls": [{"function": {"name": "dance", "arguments": {}}}]},
        )

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        return _pcm_bytes(2400)

    async def fake_start_tool_calls(tool_calls):
        return [("dance1", bg_tool)]

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("dance!")

    assert llm_call_count == 1


@pytest.mark.asyncio
async def test_tool_message_appended_to_history() -> None:
    """A role=tool message with tool_call_id appears in history before the second LLM call."""
    from robot_comic.tools.background_tool_manager import BackgroundTool, ToolState

    handler = _make_handler()
    messages_seen_on_second_call: list[dict] = []
    call_count = 0
    camera_result = {
        "description": "An older gentleman in a Hawaiian shirt waves at the camera enthusiastically."
    }

    async def instant_task() -> None:
        pass

    bg_tool = BackgroundTool(
        id="cam2",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result=camera_result,
    )
    bg_tool._task = asyncio.create_task(instant_task())

    async def patched_llm():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                "Checking the camera!",
                [{"function": {"name": "camera", "arguments": {}}}],
                {"role": "assistant", "content": "Checking the camera!",
                 "tool_calls": [{"function": {"name": "camera", "arguments": {}}}]},
            )
        messages_seen_on_second_call.extend(list(handler._conversation_history))
        return "I see someone waving!", [], {"role": "assistant", "content": "I see someone waving!"}

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        return _pcm_bytes(2400)

    async def fake_start_tool_calls(tool_calls):
        return [("cam2", bg_tool)]

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("who's there?")

    tool_messages = [m for m in messages_seen_on_second_call if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "cam2"
    import json as _json
    content = _json.loads(tool_messages[0]["content"])
    assert "description" in content


@pytest.mark.asyncio
async def test_assistant_message_preserves_tool_calls_field() -> None:
    """When Ollama returns tool_calls, the history entry has a tool_calls field."""
    handler = _make_handler()

    async def patched_llm():
        raw_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "dance", "arguments": {}}}],
        }
        return "", [{"function": {"name": "dance", "arguments": {}}}], raw_msg

    async def fake_start_tool_calls(tool_calls):
        return []

    handler._call_llm = patched_llm  # type: ignore[method-assign]
    handler._start_tool_calls = fake_start_tool_calls  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("dance!")

    assistant_entries = [m for m in handler._conversation_history if m.get("role") == "assistant"]
    assert len(assistant_entries) == 1
    assert "tool_calls" in assistant_entries[0]
    assert assistant_entries[0]["tool_calls"][0]["function"]["name"] == "dance"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_second_llm_pass_fires_on_meaningful_result tests/test_chatterbox_tts.py::test_no_second_pass_for_action_tools tests/test_chatterbox_tts.py::test_tool_message_appended_to_history tests/test_chatterbox_tts.py::test_assistant_message_preserves_tool_calls_field -v`
Expected: FAIL (second LLM pass not wired up, `_start_tool_calls` return value discarded, etc.)

- [ ] **Step 3: Extract `_synthesize_and_enqueue` helper**

Add as an instance method on `ChatterboxTTSResponseHandler`, directly before `_dispatch_completed_transcript`:

```python
async def _synthesize_and_enqueue(self, response_text: str) -> None:
    """Translate response_text to TTS segments and enqueue PCM frames."""
    if not response_text:
        return
    persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default"
    segments = translate(response_text, persona=persona, use_turbo=False)
    any_audio = False
    for seg in segments:
        if seg.silence_ms:
            for frame in self._pcm_to_frames(self._silence_pcm(seg.silence_ms)):
                await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
            any_audio = True
        else:
            text = f"{seg.turbo_insert} {seg.text}" if seg.turbo_insert else seg.text
            for sentence in _split_sentences(text):
                pcm = await self._call_chatterbox_tts(
                    sentence, exaggeration=seg.exaggeration, cfg_weight=seg.cfg_weight
                )
                if pcm:
                    for frame in self._pcm_to_frames(pcm):
                        await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
                    any_audio = True
    if not any_audio:
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": "[TTS error]"})
        )
```

- [ ] **Step 4: Replace `_dispatch_completed_transcript` with two-phase version**

Replace the entire `_dispatch_completed_transcript` method with:

```python
async def _dispatch_completed_transcript(self, transcript: str) -> None:
    """LLM → tool dispatch → TTS → PCM frames (two-phase with query-tool feedback)."""
    from robot_comic.tools.background_tool_manager import BackgroundTool

    self._conversation_history.append({"role": "user", "content": transcript})

    try:
        response_text, tool_calls, raw_message = await self._call_llm()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return

    self._conversation_history.append(raw_message)

    bg_tools: list[tuple[str, BackgroundTool]] = []
    if tool_calls:
        bg_tools = await self._start_tool_calls(tool_calls)

    await self.output_queue.put(
        AdditionalOutputs({"role": "assistant", "content": response_text})
    )
    await self._synthesize_and_enqueue(response_text)

    if not bg_tools:
        return

    tool_results = await self._await_tool_results(bg_tools)
    meaningful = {
        cid: result
        for cid, result in tool_results.items()
        if self._is_meaningful_result(result)
    }
    if not meaningful:
        return

    for call_id, result in meaningful.items():
        self._conversation_history.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": call_id,
        })

    try:
        follow_up_text, _, _ = await self._call_llm()
    except Exception as exc:
        logger.warning("Phase-2 LLM call failed: %s", exc)
        return

    self._conversation_history.append({"role": "assistant", "content": follow_up_text})
    await self.output_queue.put(
        AdditionalOutputs({"role": "assistant", "content": follow_up_text})
    )
    await self._synthesize_and_enqueue(follow_up_text)
```

- [ ] **Step 5: Run new tests to verify they pass**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py::test_second_llm_pass_fires_on_meaningful_result tests/test_chatterbox_tts.py::test_no_second_pass_for_action_tools tests/test_chatterbox_tts.py::test_tool_message_appended_to_history tests/test_chatterbox_tts.py::test_assistant_message_preserves_tool_calls_field -v`
Expected: PASS

- [ ] **Step 6: Run full suite to catch regressions**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py -v`
Expected: all 49 tests pass, 0 failures

- [ ] **Step 7: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat(chatterbox): two-phase tool result feedback with second LLM pass (#48)"
```

---

### Task 5: Final verification and close issue

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite with verbose output**

Run: `uv run python -m pytest tests/test_chatterbox_tts.py -v --tb=short`
Expected: 49 tests pass, 0 failures

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py`
Expected: no errors

- [ ] **Step 3: Run type checker**

Run: `uv run mypy src/robot_comic/chatterbox_tts.py --pretty --show-error-codes`
Expected: no new errors beyond pre-existing ones

- [ ] **Step 4: Close GitHub issue**

```bash
gh issue close 48 --comment "Implemented two-phase tool result feedback (#48). Phase 1 speaks immediately; Phase 2 awaits query tool results (5s timeout), feeds meaningful results back to Ollama, speaks the follow-up. 49 tests pass."
```
