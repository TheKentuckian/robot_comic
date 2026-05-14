# Integration Smoke Tests

## What they cover

`tests/integration/` holds **end-to-end handler lifecycle smoke tests** that
complement the ~400 unit tests already in `tests/`.

Unit tests verify individual components (sentence splitter, retry logic, config
parsing, etc.) with everything mocked.  The smoke tests verify the **wiring**:
that a handler boots, processes a synthetic input, and emits the expected output
through its queue — exercising the full dispatch chain in one shot.

Current coverage:

| Test file | Handler | What is mocked |
|-----------|---------|----------------|
| `test_chatterbox_smoke.py` | `LocalSTTChatterboxHandler` | `_http.stream` (llama-server SSE) + `_call_chatterbox_tts` |
| `test_gemini_live_smoke.py` | `GeminiLiveHandler` | `client.aio.live.connect` (Gemini Live SDK session boundary) |

## Running the tests

Integration tests are **skipped by default** (excluded via `addopts` in
`pyproject.toml`) so they do not slow down the normal CI matrix.

```bash
# Run only integration tests
pytest tests/integration/ -m integration -v

# Run everything except integration tests (the default)
pytest -q

# Run the full suite including integration
pytest -m '' -q
```

## Adding a new smoke test

1. Create `tests/integration/test_<backend>_smoke.py`.
2. Decorate every test with `@pytest.mark.integration` **and** `@pytest.mark.asyncio`.
3. Use `make_tool_deps()` from `conftest.py` to get a fully mocked
   `ToolDependencies` (no real robot required).
4. Use `drain_queue(handler.output_queue)` to collect output items after the
   dispatch call.
5. Mock at the **transport boundary** only — the HTTP client (`_http.stream`,
   `_http.post`) for llama-server backends, or the SDK session object for
   Gemini Live — so that the real routing, splitting, and queue-push logic is
   exercised.

### Minimal template

```python
import pytest
from .conftest import drain_queue, make_tool_deps

@pytest.mark.integration
@pytest.mark.asyncio
async def test_my_handler_produces_audio() -> None:
    from robot_comic.my_handler import MyHandler
    deps = make_tool_deps()
    handler = MyHandler(deps, sim_mode=True)
    handler._http = AsyncMock()
    handler._http.stream = MagicMock(return_value=_make_sse_stream([...]))

    await handler._dispatch_completed_transcript("hello")

    items = drain_queue(handler.output_queue)
    audio_frames = [i for i in items if isinstance(i, tuple)]
    assert len(audio_frames) >= 1
```

## Design principles

- **Hermetic**: no real network, no real robot, no filesystem reads beyond
  what the handler itself performs at init time.
- **Fast**: each test should complete in under 2 seconds.
- **Narrow mock surface**: only the transport boundary is mocked; everything
  else (sentence splitting, PCM chunking, queue wiring) runs real code.
