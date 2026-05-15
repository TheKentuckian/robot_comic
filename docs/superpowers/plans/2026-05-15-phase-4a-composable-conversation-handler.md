# Phase 4a Implementation Plan — `ComposableConversationHandler`

**Goal:** Wrap `ComposablePipeline` in a `ConversationHandler` ABC implementation so 4b can route the factory through it.

**Architecture:** One new class in one new file. Holds a `ComposablePipeline` + a legacy TTS handler (for voice/personality forwarding) + a `build` closure (for `copy()`). All nine ABC methods implemented as thin delegates except `apply_personality` (resets history + re-seeds system prompt).

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, fastrtc `wait_for_item`.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4a-composable-conversation-handler.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/composable_conversation_handler.py` | NEW — `ComposableConversationHandler` class |
| `tests/test_composable_conversation_handler.py` | NEW — all unit + integration tests |

No changes elsewhere.

---

### Task 1: Skeleton class — ABC implementation, no-op bodies

**Files:**
- Create: `src/robot_comic/composable_conversation_handler.py`
- Test: `tests/test_composable_conversation_handler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composable_conversation_handler.py
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic.conversation_handler import ConversationHandler
from robot_comic.composable_conversation_handler import ComposableConversationHandler


def _make_wrapper() -> ComposableConversationHandler:
    pipeline = MagicMock()
    pipeline.output_queue = asyncio.Queue()
    pipeline._conversation_history = []
    pipeline.start_up = AsyncMock()
    pipeline.shutdown = AsyncMock()
    pipeline.feed_audio = AsyncMock()
    pipeline.reset_history = MagicMock()

    tts_handler = MagicMock()
    deps = MagicMock()

    def _build() -> ComposableConversationHandler:
        return _make_wrapper()

    return ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=tts_handler,
        deps=deps,
        build=_build,
    )


def test_wrapper_implements_conversation_handler_abc() -> None:
    wrapper = _make_wrapper()
    assert isinstance(wrapper, ConversationHandler)
```

Run: `/venvs/apps_venv/bin/python -m pytest tests/test_composable_conversation_handler.py::test_wrapper_implements_conversation_handler_abc -v`

Expected: `ImportError` or `ModuleNotFoundError` — the class doesn't exist.

- [ ] **Step 2: Minimum implementation**

```python
# src/robot_comic/composable_conversation_handler.py
"""ConversationHandler ABC wrapper around ComposablePipeline (Phase 4a).

Closes the surface gap between ``ComposablePipeline`` and the existing
``ConversationHandler`` ABC so the factory (Phase 4b) can return either
interchangeably. Forwards voice / personality calls to a legacy TTS
handler held by reference — no Protocol churn.
"""

from __future__ import annotations
import logging
from typing import Any, Callable

from robot_comic.composable_pipeline import ComposablePipeline
from robot_comic.conversation_handler import AudioFrame, ConversationHandler, HandlerOutput
from robot_comic.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


class ComposableConversationHandler(ConversationHandler):
    """Wrap a ``ComposablePipeline`` as a ``ConversationHandler``."""

    def __init__(
        self,
        pipeline: ComposablePipeline,
        *,
        tts_handler: ConversationHandler,
        deps: ToolDependencies,
        build: Callable[[], "ComposableConversationHandler"],
    ) -> None:
        self.pipeline = pipeline
        self._tts_handler = tts_handler
        self.deps = deps
        self._build = build
        self.output_queue = pipeline.output_queue
        self._clear_queue = None

    def copy(self) -> "ComposableConversationHandler":
        raise NotImplementedError

    async def start_up(self) -> None:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError

    async def receive(self, frame: AudioFrame) -> None:
        raise NotImplementedError

    async def emit(self) -> HandlerOutput:
        raise NotImplementedError

    async def apply_personality(self, profile: str | None) -> str:
        raise NotImplementedError

    async def get_available_voices(self) -> list[str]:
        raise NotImplementedError

    def get_current_voice(self) -> str:
        raise NotImplementedError

    async def change_voice(self, voice: str) -> str:
        raise NotImplementedError
```

Run test again. Expected: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py
git commit -m "feat(handler): Phase 4a skeleton ComposableConversationHandler (#337)"
```

---

### Task 2: Lifecycle methods — `start_up`, `shutdown`, `receive`, `emit`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_composable_conversation_handler.py`:

```python
@pytest.mark.asyncio
async def test_start_up_delegates_to_pipeline() -> None:
    wrapper = _make_wrapper()
    await wrapper.start_up()
    wrapper.pipeline.start_up.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_delegates_to_pipeline() -> None:
    wrapper = _make_wrapper()
    await wrapper.shutdown()
    wrapper.pipeline.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_forwards_to_feed_audio() -> None:
    wrapper = _make_wrapper()
    frame: AudioFrame = (16000, np.zeros(160, dtype=np.int16))
    await wrapper.receive(frame)
    wrapper.pipeline.feed_audio.assert_awaited_once_with(frame)


@pytest.mark.asyncio
async def test_emit_pulls_from_output_queue() -> None:
    wrapper = _make_wrapper()
    sentinel = (24000, np.ones(48, dtype=np.int16))
    await wrapper.pipeline.output_queue.put(sentinel)
    result = await wrapper.emit()
    assert result is sentinel
```

Add imports at file top: `import numpy as np`, `from robot_comic.conversation_handler import AudioFrame`.

Run: all four fail with `NotImplementedError`.

- [ ] **Step 2: Implement the four methods**

Replace stubs in `composable_conversation_handler.py`:

```python
async def start_up(self) -> None:
    """Delegate to ``ComposablePipeline.start_up`` — blocks until shutdown."""
    await self.pipeline.start_up()

async def shutdown(self) -> None:
    """Delegate to ``ComposablePipeline.shutdown``."""
    await self.pipeline.shutdown()

async def receive(self, frame: AudioFrame) -> None:
    """Forward a captured input frame to the pipeline's STT backend."""
    await self.pipeline.feed_audio(frame)

async def emit(self) -> HandlerOutput:
    """Pull the next output item from the pipeline's output queue."""
    from fastrtc import wait_for_item  # deferred — fastrtc pulls gradio at boot

    return await wait_for_item(self.pipeline.output_queue)
```

Run tests. The first three should pass. `test_emit_pulls_from_output_queue` may pass or fail depending on `wait_for_item`'s behaviour against a pre-populated queue — if it fails, replace `wait_for_item` with `self.pipeline.output_queue.get()` and re-run the existing handlers' tests to confirm `get()` is acceptable (the existing `emit` in `elevenlabs_tts.py:464` does post-processing inside `emit`, but the wrapper does not).

Expected: green.

- [ ] **Step 3: Commit**

```
git add -u
git commit -m "feat(handler): wire lifecycle + audio I/O on ComposableConversationHandler (#337)"
```

---

### Task 3: Voice forwarding — `get_current_voice`, `get_available_voices`, `change_voice`

- [ ] **Step 1: Write the failing tests**

```python
def test_get_current_voice_delegates() -> None:
    wrapper = _make_wrapper()
    wrapper._tts_handler.get_current_voice = MagicMock(return_value="Brian")
    assert wrapper.get_current_voice() == "Brian"
    wrapper._tts_handler.get_current_voice.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_voices_delegates() -> None:
    wrapper = _make_wrapper()
    wrapper._tts_handler.get_available_voices = AsyncMock(return_value=["A", "B"])
    assert await wrapper.get_available_voices() == ["A", "B"]
    wrapper._tts_handler.get_available_voices.assert_awaited_once()


@pytest.mark.asyncio
async def test_change_voice_delegates() -> None:
    wrapper = _make_wrapper()
    wrapper._tts_handler.change_voice = AsyncMock(return_value="Voice changed to X.")
    assert await wrapper.change_voice("X") == "Voice changed to X."
    wrapper._tts_handler.change_voice.assert_awaited_once_with("X")
```

Run: three failures with `NotImplementedError`.

- [ ] **Step 2: Implement the three delegates**

```python
async def get_available_voices(self) -> list[str]:
    return await self._tts_handler.get_available_voices()

def get_current_voice(self) -> str:
    return self._tts_handler.get_current_voice()

async def change_voice(self, voice: str) -> str:
    return await self._tts_handler.change_voice(voice)
```

Run tests. Expected: green.

- [ ] **Step 3: Commit**

```
git add -u
git commit -m "feat(handler): forward voice methods to legacy TTS handler (#337)"
```

---

### Task 4: `apply_personality` — reset history + re-seed system prompt

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_apply_personality_resets_history_and_reseeds(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _make_wrapper()
    # Seed some history that should be wiped.
    wrapper.pipeline._conversation_history = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]
    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.set_custom_profile",
        lambda profile: None,
    )
    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.get_session_instructions",
        lambda: "fresh instructions",
    )

    result = await wrapper.apply_personality("rodney")

    assert "Applied personality 'rodney'" in result
    wrapper.pipeline.reset_history.assert_called_once_with(keep_system=False)
    # Wrapper re-seeds via direct mutation post-reset
    assert wrapper.pipeline._conversation_history[-1] == {
        "role": "system",
        "content": "fresh instructions",
    }


@pytest.mark.asyncio
async def test_apply_personality_returns_failure_message_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _make_wrapper()

    def _boom(profile: str | None) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(
        "robot_comic.composable_conversation_handler.set_custom_profile", _boom
    )

    result = await wrapper.apply_personality("broken")

    assert "Failed to apply personality" in result
    assert "nope" in result
    wrapper.pipeline.reset_history.assert_not_called()
```

Note: the first test's reset-then-mutate assertion requires the implementation to call `reset_history(keep_system=False)` first and then append the new system message. To make the assertion work against the MagicMock `reset_history` (which doesn't actually mutate `_conversation_history`), pre-seed `_conversation_history = []` before the append assertion. Adjust test setup accordingly.

Run: two failures.

- [ ] **Step 2: Implement `apply_personality`**

Add imports at top:

```python
from robot_comic.config import get_session_instructions, set_custom_profile
```

(Verify these names. `set_custom_profile` is imported by `elevenlabs_tts.py`; `get_session_instructions` by `local_stt_realtime.py:893`. If they live in a different module on `main`, adjust.)

Replace stub:

```python
async def apply_personality(self, profile: str | None) -> str:
    """Switch personality, reset pipeline history, and re-seed system prompt."""
    try:
        set_custom_profile(profile)
    except Exception as exc:
        logger.error("Error applying personality %r: %s", profile, exc)
        return f"Failed to apply personality: {exc}"
    self.pipeline.reset_history(keep_system=False)
    self.pipeline._conversation_history.append(
        {"role": "system", "content": get_session_instructions()}
    )
    return f"Applied personality {profile!r}. Conversation history reset."
```

Run tests. Expected: green.

- [ ] **Step 3: Commit**

```
git add -u
git commit -m "feat(handler): apply_personality resets history and reseeds system prompt (#337)"
```

---

### Task 5: `copy` — fresh wrapper via build closure

- [ ] **Step 1: Write the failing tests**

```python
def test_copy_returns_new_instance_from_build_closure() -> None:
    build_count = {"n": 0}
    sentinel_pipeline = MagicMock()
    sentinel_pipeline.output_queue = asyncio.Queue()

    def _build() -> ComposableConversationHandler:
        build_count["n"] += 1
        return ComposableConversationHandler(
            pipeline=sentinel_pipeline,
            tts_handler=MagicMock(),
            deps=MagicMock(),
            build=_build,
        )

    original = _build()
    build_count["n"] = 0  # reset after construction
    copy = original.copy()
    assert copy is not original
    assert build_count["n"] == 1


def test_copy_does_not_share_pipeline_state() -> None:
    # Build closure that creates real ComposablePipeline-shaped fakes.
    def _make_fake() -> MagicMock:
        m = MagicMock()
        m.output_queue = asyncio.Queue()
        m._conversation_history = []
        return m

    def _build() -> ComposableConversationHandler:
        return ComposableConversationHandler(
            pipeline=_make_fake(),
            tts_handler=MagicMock(),
            deps=MagicMock(),
            build=_build,
        )

    original = _build()
    copy = original.copy()

    original.pipeline._conversation_history.append({"role": "user", "content": "hi"})
    assert copy.pipeline._conversation_history == []
```

Run: failures with `NotImplementedError`.

- [ ] **Step 2: Implement `copy`**

```python
def copy(self) -> "ComposableConversationHandler":
    """Build a fresh wrapper + pipeline via the injected factory closure."""
    return self._build()
```

Run tests. Expected: green.

- [ ] **Step 3: Commit**

```
git add -u
git commit -m "feat(handler): implement copy via injected build closure (#337)"
```

---

### Task 6: Integration test — full transcript → audio frame round trip

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_integration_transcript_to_audio_frame() -> None:
    """End-to-end: receive frame -> mocked STT completes -> mocked LLM -> mocked TTS -> emit."""
    from robot_comic.backends import LLMResponse
    from robot_comic.composable_pipeline import ComposablePipeline

    on_completed_cb: dict[str, Any] = {}

    class StubSTT:
        async def start(self, on_completed: Any) -> None:
            on_completed_cb["fn"] = on_completed

        async def feed_audio(self, frame: AudioFrame) -> None:
            pass

        async def stop(self) -> None:
            pass

    class StubLLM:
        async def prepare(self) -> None:
            pass

        async def chat(self, history: list[Any], tools: Any) -> LLMResponse:
            return LLMResponse(text="hello world", tool_calls=())

        async def shutdown(self) -> None:
            pass

    class StubTTS:
        async def prepare(self) -> None:
            pass

        async def synthesize(self, text: str, tags: Any = ()) -> Any:
            yield (24000, np.ones(48, dtype=np.int16))

        async def shutdown(self) -> None:
            pass

    pipeline = ComposablePipeline(StubSTT(), StubLLM(), StubTTS())

    def _build() -> ComposableConversationHandler:
        raise AssertionError("copy not exercised here")

    wrapper = ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=MagicMock(),
        deps=MagicMock(),
        build=_build,
    )

    # Bring the pipeline online enough for the STT callback to register.
    start_task = asyncio.create_task(wrapper.start_up())
    await asyncio.sleep(0)  # let prepare()/start() run
    # Drive a "completed transcript" through the registered callback.
    assert "fn" in on_completed_cb
    await on_completed_cb["fn"]("hello")
    # The TTS frame should now be on the queue.
    frame = await asyncio.wait_for(wrapper.emit(), timeout=1.0)
    sample_rate, samples = frame
    assert sample_rate == 24000
    assert samples.shape == (48,)

    await wrapper.shutdown()
    await start_task
```

Run: expect a failure (or pass — depending on `wait_for_item` semantics against a real queue). Iterate test until it's a stable failing test, then implementation.

- [ ] **Step 2: No new implementation expected**

The earlier tasks should make this test pass. If `emit` returns the frame correctly, no code change needed. If not, diagnose `wait_for_item` vs `Queue.get()` behavior and decide which to use.

- [ ] **Step 3: Commit**

```
git add -u
git commit -m "test(handler): integration test for ComposableConversationHandler round trip (#337)"
```

---

### Task 7: Add `# TODO(phase4-lifecycle): ...` markers for deferred hooks

The wrapper does NOT plumb telemetry, supporting events, joke history, history-trim, or echo guard. Each of these is its own follow-up PR. Insert markers at the integration points so they're easy to find when the follow-ups land.

- [ ] **Step 1: Add markers (no test)**

In `composable_conversation_handler.py`, at the top of `start_up` and `apply_personality`:

```python
async def start_up(self) -> None:
    # TODO(phase4-lifecycle): emit the four supporting events from boot timeline (#321)
    # before delegating to pipeline.start_up — current behavior drops these on the
    # floor for composable mode.
    await self.pipeline.start_up()

async def apply_personality(self, profile: str | None) -> str:
    # TODO(phase4-lifecycle): the legacy handlers also clear joke history and
    # the per-session echo-guard state on persona switch. The composable pipeline
    # has neither yet — wire them when those hooks land in follow-up PRs.
    ...
```

And one at file top documenting the deferred hooks:

```python
# Phase 4a deliberately leaves these lifecycle hooks unwired; each is a
# follow-up PR between 4b and 4d:
#   - telemetry.record_llm_duration
#   - boot-timeline supporting events (#321)
#   - record_joke_history (llama_base.py:553-568)
#   - history_trim.trim_history_in_place
#   - _speaking_until echo-guard timestamps (elevenlabs_tts.py:471-473)
```

- [ ] **Step 2: Commit**

```
git add -u
git commit -m "docs(handler): flag deferred lifecycle hooks in ComposableConversationHandler (#337)"
```

---

### Task 8: Pre-flight — ruff, mypy, full pytest

- [ ] **Step 1: Lint and format**

```
ruff check src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py --fix
ruff format src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py
```

- [ ] **Step 2: Type-check**

```
mypy --pretty --show-error-codes src/robot_comic/composable_conversation_handler.py
```

- [ ] **Step 3: Full test suite**

```
/venvs/apps_venv/bin/python -m pytest tests/ -x
```

Expected: green. Any unexpected failure means the wrapper unintentionally affected an existing path — investigate before pushing.

- [ ] **Step 4: Commit if any lint/format fixes happened**

```
git add -u
git commit -m "style: ruff fixes on Phase 4a wrapper (#337)"
```

(Skip if no diff.)

---

### Task 9: Push branch, open PR

- [ ] **Step 1: Push**

```
git push -u origin claude/phase-4-composable-orchestrator-maQ4J
```

- [ ] **Step 2: Open PR** (only when operator explicitly requests one — per the harness instructions, do not open a PR automatically)

PR title: `feat(handler): Phase 4a — ComposableConversationHandler wrapper (#337)`

PR body should include:
- Link to the spec doc
- Bullet list of the ABC methods now implemented
- Explicit "no factory routing changes — 4b will wire this in" disclaimer
- TODO list of the deferred lifecycle hooks

---

## Notes for the agent executing this plan

- Use the central venv `/venvs/apps_venv` per CLAUDE.md, not a local `.venv`.
- Each task ends with a green test and a commit. Do not stack tasks into one commit — each is independently revertible.
- The `apply_personality` test in Task 4 has a subtle setup issue (MagicMock `reset_history` doesn't actually mutate the list). Adjust the test setup so the assertion is meaningful — either patch `reset_history` to clear the list, or seed an empty list and only assert the post-reset append.
- If `fastrtc.wait_for_item` doesn't work the way the existing handlers use it (race conditions on a freshly-populated queue), fall back to `await self.pipeline.output_queue.get()` and document why in a comment.
- Do NOT change the factory in this sub-phase. If a test seems to need factory changes, the test is in the wrong sub-phase — defer it to 4b.
