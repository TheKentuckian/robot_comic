# Hermes3 Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `ChatterboxTTSResponseHandler._call_llm()` so Hermes3 uses structured tool calls consistently, with a self-correcting nudge when it doesn't.

**Architecture:** Three static improvements (prompt addendum, JSON-in-content parser, arg parser fix) plus one dynamic recovery (ephemeral retry-with-nudge on semantically empty responses). All changes are confined to `chatterbox_tts.py`; nudge logic lives in a new `_nudge_llm()` method.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, Ollama `/api/chat`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/chatterbox_tts.py` | All production changes |
| `tests/test_chatterbox_tts.py` | All new tests (append to existing file) |

---

### Task 1: Fix `_parse_text_tool_args` — JSON-first arg parsing

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py` (lines 61–69 — `_parse_text_tool_args`)
- Test: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chatterbox_tts.py`:

```python
# ---------------------------------------------------------------------------
# _parse_text_tool_args
# ---------------------------------------------------------------------------

def test_parse_text_tool_args_json_dict() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args('{"action": "scan", "name": "Alice"}')
    assert result == {"action": "scan", "name": "Alice"}


def test_parse_text_tool_args_value_with_comma() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args('{"message": "hello, world"}')
    assert result == {"message": "hello, world"}


def test_parse_text_tool_args_bare_kv_fallback() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args("action: scan")
    assert result == {"action": "scan"}


def test_parse_text_tool_args_empty_string() -> None:
    from robot_comic.chatterbox_tts import _parse_text_tool_args

    result = _parse_text_tool_args("")
    assert result == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_chatterbox_tts.py::test_parse_text_tool_args_json_dict tests/test_chatterbox_tts.py::test_parse_text_tool_args_value_with_comma -v
```

Expected: FAIL — current implementation splits on `,` and won't handle JSON.

- [ ] **Step 3: Replace `_parse_text_tool_args` in `chatterbox_tts.py`**

Replace the existing function (lines 61–69) with:

```python
def _parse_text_tool_args(kv_str: str) -> dict[str, Any]:
    """Parse args from a Hermes3 text-format tool call.

    Tries json.loads() first (handles quoted values and commas in values),
    then falls back to bare key:value comma-split.
    """
    kv_str = kv_str.strip()
    if kv_str:
        try:
            parsed = json.loads(kv_str)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    args: dict[str, Any] = {}
    for pair in kv_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, _, v = pair.partition(":")
            args[k.strip()] = v.strip()
    return args
```

Note: return type changes from `dict[str, str]` to `dict[str, Any]` to accommodate JSON-parsed values.

- [ ] **Step 4: Run all four new tests**

```
uv run pytest tests/test_chatterbox_tts.py::test_parse_text_tool_args_json_dict tests/test_chatterbox_tts.py::test_parse_text_tool_args_value_with_comma tests/test_chatterbox_tts.py::test_parse_text_tool_args_bare_kv_fallback tests/test_chatterbox_tts.py::test_parse_text_tool_args_empty_string -v
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```
uv run pytest tests/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "fix: _parse_text_tool_args tries json.loads before comma-split"
```

---

### Task 2: Add JSON-in-content tool call detection

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py` — add `_parse_json_content_tool_call()` helper and wire it into `_call_llm()`
- Test: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chatterbox_tts.py`:

```python
# ---------------------------------------------------------------------------
# _parse_json_content_tool_call
# ---------------------------------------------------------------------------

def test_parse_json_content_openai_style() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    text = '{"function": {"name": "greet", "arguments": {"action": "scan"}}}'
    result = _parse_json_content_tool_call(text)
    assert result == ("greet", {"action": "scan"})


def test_parse_json_content_flat_style() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    text = '{"name": "play_emotion", "arguments": {"emotion": "laughing1"}}'
    result = _parse_json_content_tool_call(text)
    assert result == ("play_emotion", {"emotion": "laughing1"})


def test_parse_json_content_flat_style_no_arguments_key() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    # {"name": "greet"} with no arguments key — returns empty dict for args
    text = '{"name": "greet"}'
    result = _parse_json_content_tool_call(text)
    assert result == ("greet", {})


def test_parse_json_content_returns_none_for_plain_text() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    assert _parse_json_content_tool_call("Hello, how are you?") is None


def test_parse_json_content_returns_none_for_invalid_json() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    assert _parse_json_content_tool_call("{not valid json}") is None


def test_parse_json_content_returns_none_for_json_without_name() -> None:
    from robot_comic.chatterbox_tts import _parse_json_content_tool_call

    assert _parse_json_content_tool_call('{"foo": "bar"}') is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_chatterbox_tts.py::test_parse_json_content_openai_style tests/test_chatterbox_tts.py::test_parse_json_content_flat_style -v
```

Expected: FAIL — `_parse_json_content_tool_call` does not exist yet.

- [ ] **Step 3: Add the helper function to `chatterbox_tts.py`**

Add after `_parse_text_tool_args` (before `_split_sentences`):

```python
def _parse_json_content_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Try to extract a tool call from JSON-formatted content text.

    Handles two shapes Hermes3 may emit:
      OpenAI-style: {"function": {"name": "...", "arguments": {...}}}
      Flat-style:   {"name": "...", "arguments": {...}}

    Returns (fn_name, args) or None if the text is not a recognisable tool call.
    """
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    # OpenAI-style: {"function": {"name": "...", "arguments": {...}}}
    fn = data.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if name and isinstance(name, str):
            args = fn.get("arguments") or {}
            return name, args if isinstance(args, dict) else {}

    # Flat-style: {"name": "...", "arguments": {...}}
    name = data.get("name")
    if name and isinstance(name, str):
        args = data.get("arguments") or {}
        return name, args if isinstance(args, dict) else {}

    return None
```

- [ ] **Step 4: Wire into `_call_llm()` — add JSON-in-content check after the regex check**

In `_call_llm()`, find the block (around line 382–394):

```python
                # Hermes3 sometimes puts tool calls as plain text instead of tool_calls
                if not tool_calls and text:
                    m = _TEXT_TOOL_CALL_RE.match(text)
                    if m:
                        fn_name = m.group(1)
                        args = _parse_text_tool_args(m.group(2) or "")
                        tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                        logger.warning(
                            "Hermes3 text-format tool call in content field: %s(%r) — dispatching and suppressing TTS",
                            fn_name, args,
                        )
                        text = ""
```

Replace with:

```python
                # Hermes3 sometimes puts tool calls as plain text instead of tool_calls
                if not tool_calls and text:
                    m = _TEXT_TOOL_CALL_RE.match(text)
                    if m:
                        fn_name = m.group(1)
                        args = _parse_text_tool_args(m.group(2) or "")
                        tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                        logger.warning(
                            "Hermes3 text-format tool call in content field: %s(%r) — dispatching and suppressing TTS",
                            fn_name, args,
                        )
                        text = ""

                # Hermes3 may also emit JSON-format tool calls in the content field
                if not tool_calls and text:
                    json_tc = _parse_json_content_tool_call(text)
                    if json_tc is not None:
                        fn_name, args = json_tc
                        tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                        logger.warning(
                            "Hermes3 JSON-format tool call in content field: %s(%r) — dispatching and suppressing TTS",
                            fn_name, args,
                        )
                        text = ""
```

- [ ] **Step 5: Write integration test for JSON-in-content going through `_call_llm`**

Append to `tests/test_chatterbox_tts.py`:

```python
@pytest.mark.asyncio
async def test_call_llm_detects_json_content_tool_call() -> None:
    """_call_llm dispatches a JSON-format tool call found in the content field."""
    import httpx

    handler = _make_handler()

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "message": {
            "content": '{"function": {"name": "greet", "arguments": {"action": "scan"}}}',
            "tool_calls": [],
        }
    }
    handler._http.post = AsyncMock(return_value=fake_resp)

    text, tool_calls = await handler._call_llm()

    assert text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "greet"
    assert tool_calls[0]["function"]["arguments"] == {"action": "scan"}
```

- [ ] **Step 6: Run all new tests**

```
uv run pytest tests/test_chatterbox_tts.py::test_parse_json_content_openai_style tests/test_chatterbox_tts.py::test_parse_json_content_flat_style tests/test_chatterbox_tts.py::test_parse_json_content_flat_style_no_arguments_key tests/test_chatterbox_tts.py::test_parse_json_content_returns_none_for_plain_text tests/test_chatterbox_tts.py::test_parse_json_content_returns_none_for_invalid_json tests/test_chatterbox_tts.py::test_parse_json_content_returns_none_for_json_without_name tests/test_chatterbox_tts.py::test_call_llm_detects_json_content_tool_call -v
```

Expected: all PASS.

- [ ] **Step 7: Run full test suite**

```
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat: detect JSON-format tool calls in Hermes3 content field"
```

---

### Task 3: Inject tool-use addendum into system prompt

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py` — add module-level `_TOOL_USE_ADDENDUM` constant and apply in `_call_llm()`
- Test: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatterbox_tts.py`:

```python
@pytest.mark.asyncio
async def test_call_llm_injects_tool_use_addendum() -> None:
    """The system message sent to Ollama includes the tool-use addendum."""
    import httpx
    from robot_comic.chatterbox_tts import _TOOL_USE_ADDENDUM

    handler = _make_handler()
    captured_payloads: list[dict] = []

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"message": {"content": "Hi!", "tool_calls": []}}

    async def capturing_post(url, *, json=None, **kwargs):
        captured_payloads.append(json or {})
        return fake_resp

    handler._http.post = capturing_post  # type: ignore[method-assign]

    await handler._call_llm()

    assert captured_payloads, "No LLM call was made"
    messages = captured_payloads[0]["messages"]
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert _TOOL_USE_ADDENDUM in system_content
```

- [ ] **Step 2: Run test to confirm it fails**

```
uv run pytest tests/test_chatterbox_tts.py::test_call_llm_injects_tool_use_addendum -v
```

Expected: FAIL — `_TOOL_USE_ADDENDUM` does not exist yet.

- [ ] **Step 3: Add `_TOOL_USE_ADDENDUM` constant and apply it in `_call_llm()`**

Add this constant near the top of `chatterbox_tts.py`, after the existing module-level regex constants:

```python
_TOOL_USE_ADDENDUM = (
    "\n\n## TOOL CALL RULES\n"
    "Always invoke tools using the structured tool_calls mechanism — never embed tool calls as text.\n"
    "When a tool call is required, emit only the tool call; do not add explanatory prose alongside it.\n"
    "Never write {function: name, ...} or any text representation of a tool call."
)
```

In `_call_llm()`, change the line that reads:

```python
        system_prompt = get_session_instructions()
```

to:

```python
        system_prompt = get_session_instructions() + _TOOL_USE_ADDENDUM
```

- [ ] **Step 4: Run the test**

```
uv run pytest tests/test_chatterbox_tts.py::test_call_llm_injects_tool_use_addendum -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat: inject tool-use addendum into Hermes3 system prompt"
```

---

### Task 4: Retry-with-nudge for semantically empty responses

**Files:**
- Modify: `src/robot_comic/chatterbox_tts.py` — add `_nudge_llm()` method, wire into `_call_llm()`
- Test: `tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chatterbox_tts.py`:

```python
# ---------------------------------------------------------------------------
# Retry-with-nudge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nudge_fires_on_empty_response() -> None:
    """When LLM returns empty text and no tool calls, a nudge is sent and its tool call returned."""
    handler = _make_handler()
    nudge_detected = False

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal nudge_detected
        msgs = json.get("messages", [])
        if any(m.get("content") == "Please use a tool call now." for m in msgs):
            nudge_detected = True
            resp_data = {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "greet", "arguments": {"action": "scan"}}}],
                }
            }
        else:
            resp_data = {"message": {"content": "", "tool_calls": []}}
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = resp_data
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls = await handler._call_llm()

    assert nudge_detected
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "greet"


@pytest.mark.asyncio
async def test_nudge_fires_at_most_once() -> None:
    """Even if nudge response is also empty, only one nudge is sent."""
    handler = _make_handler()
    nudge_count = 0

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal nudge_count
        msgs = json.get("messages", [])
        if any(m.get("content") == "Please use a tool call now." for m in msgs):
            nudge_count += 1
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "", "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls = await handler._call_llm()

    assert nudge_count == 1
    assert text == ""
    assert tool_calls == []


@pytest.mark.asyncio
async def test_nudge_not_fired_when_text_present() -> None:
    """No nudge when the response contains meaningful text."""
    handler = _make_handler()
    call_count = 0

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "Hello there!", "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    text, tool_calls = await handler._call_llm()

    assert call_count == 1
    assert text == "Hello there!"


@pytest.mark.asyncio
async def test_nudge_not_fired_when_tool_calls_present() -> None:
    """No nudge when the response already contains tool calls."""
    handler = _make_handler()
    call_count = 0

    async def fake_post(url, *, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "greet", "arguments": {}}}],
            }
        }
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    _, tool_calls = await handler._call_llm()

    assert call_count == 1
    assert len(tool_calls) == 1


@pytest.mark.asyncio
async def test_nudge_does_not_modify_conversation_history() -> None:
    """The nudge message is ephemeral — not saved to _conversation_history."""
    handler = _make_handler()
    handler._conversation_history = [{"role": "user", "content": "hello"}]
    history_before = list(handler._conversation_history)

    async def fake_post(url, *, json=None, **kwargs):
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "", "tool_calls": []}}
        return fake_resp

    handler._http.post = fake_post  # type: ignore[method-assign]
    await handler._call_llm()

    assert handler._conversation_history == history_before
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_chatterbox_tts.py::test_nudge_fires_on_empty_response tests/test_chatterbox_tts.py::test_nudge_fires_at_most_once -v
```

Expected: FAIL — `_nudge_llm` does not exist yet.

- [ ] **Step 3: Add `_nudge_llm()` method to `ChatterboxTTSResponseHandler`**

Add this method after `_call_llm()` in `ChatterboxTTSResponseHandler`:

```python
    async def _nudge_llm(
        self,
        original_messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """One ephemeral retry with a tool-use nudge appended to the conversation.

        The nudge messages are not saved to _conversation_history.
        Applies the same text-format and JSON-in-content detection as _call_llm().
        """
        assert self._http is not None
        nudge_messages = original_messages + [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Please use a tool call now."},
        ]
        try:
            r = await self._http.post(
                f"{self._ollama_base_url}/api/chat",
                json={**payload, "messages": nudge_messages},
            )
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {})
            text = (msg.get("content") or "").strip()
            tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

            if not tool_calls and text:
                m = _TEXT_TOOL_CALL_RE.match(text)
                if m:
                    fn_name = m.group(1)
                    args = _parse_text_tool_args(m.group(2) or "")
                    tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                    text = ""

            if not tool_calls and text:
                json_tc = _parse_json_content_tool_call(text)
                if json_tc is not None:
                    fn_name, args = json_tc
                    tool_calls = [{"function": {"name": fn_name, "arguments": args}}]
                    text = ""

            if not tool_calls and not text:
                logger.warning("Hermes3 still empty after nudge — skipping turn")
            else:
                logger.info(
                    "Nudge recovered: text=%r tool_calls=%d",
                    text[:60] if text else "",
                    len(tool_calls),
                )
            return text, tool_calls
        except Exception as exc:
            logger.warning("Nudge LLM call failed: %s", exc)
            return "", []
```

- [ ] **Step 4: Wire nudge into `_call_llm()`**

In `_call_llm()`, after building `messages` and `payload` (before the retry loop), add:

```python
        nudge_attempted = False
```

Then inside the retry loop, after the existing text-format and JSON-in-content detection blocks, replace:

```python
                return text, tool_calls
```

with:

```python
                if not tool_calls and not text and not nudge_attempted:
                    nudge_attempted = True
                    logger.info("Hermes3 returned empty response — attempting nudge")
                    text, tool_calls = await self._nudge_llm(messages, payload)

                return text, tool_calls
```

The full updated `_call_llm()` return path (showing context around the change) looks like:

```python
                # ... (existing text-format regex check) ...
                # ... (existing JSON-in-content check) ...

                if not tool_calls and not text and not nudge_attempted:
                    nudge_attempted = True
                    logger.info("Hermes3 returned empty response — attempting nudge")
                    text, tool_calls = await self._nudge_llm(messages, payload)

                return text, tool_calls
```

- [ ] **Step 5: Run all new nudge tests**

```
uv run pytest tests/test_chatterbox_tts.py::test_nudge_fires_on_empty_response tests/test_chatterbox_tts.py::test_nudge_fires_at_most_once tests/test_chatterbox_tts.py::test_nudge_not_fired_when_text_present tests/test_chatterbox_tts.py::test_nudge_not_fired_when_tool_calls_present tests/test_chatterbox_tts.py::test_nudge_does_not_modify_conversation_history -v
```

Expected: all PASS.

- [ ] **Step 6: Run full test suite**

```
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/robot_comic/chatterbox_tts.py tests/test_chatterbox_tts.py
git commit -m "feat: retry-with-nudge when Hermes3 returns empty response (#52)"
```
