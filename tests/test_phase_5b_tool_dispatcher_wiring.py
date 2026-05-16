"""Phase 5b regression tests — tool_dispatcher wiring on composable triples.

These tests pin the §2.2 latent-bug fix from
``docs/superpowers/specs/2026-05-16-phase-5-exploration.md``:

  > In production, ``tool_dispatcher`` is always ``None`` — every factory
  > builder in ``handler_factory.py`` constructs ``ComposablePipeline(...)``
  > with no dispatcher argument. When the LLM emits ``tool_calls``, the
  > orchestrator's tool-call branch (composable_pipeline.py:225-231) hits
  > the ``if self.tool_dispatcher is None`` warning and **breaks the loop
  > without speaking**.

Affected composable triples (those whose LLM adapter forwards
llama-server-emitted ``tool_calls`` into ``LLMResponse.tool_calls``):

* ``(moonshine, llama, elevenlabs)`` — ``LlamaLLMAdapter``
* ``(moonshine, llama, chatterbox)`` — ``LlamaLLMAdapter``
* ``(moonshine, gemini, chatterbox)`` — ``GeminiLLMAdapter``
* ``(moonshine, gemini, elevenlabs)`` — ``GeminiLLMAdapter``

The bundled-Gemini triple ``(moonshine, gemini-bundled, gemini_tts)``
dispatches tools internally via ``GeminiBundledLLMAdapter._run_llm_with_tools``;
its dispatcher attribute may legitimately stay ``None`` because no caller
exercises it on that path. It is excluded from the wiring assertion below.

Test shape — surgical, not hardware:

1. Build each composable handler via the factory (with mocked deps).
2. Assert ``pipeline.tool_dispatcher is not None`` (the bug witness).
3. Invoke the dispatcher with a fake ``ToolCall`` and assert it routes
   through ``dispatch_tool_call`` so a real LLM-emitted call would land
   on the tool layer.
4. Assert the dispatcher result is a string (the Protocol's contract — the
   orchestrator appends it to history as a ``role=tool`` message).
"""

from __future__ import annotations
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from robot_comic.backends import ToolCall
from robot_comic.config import (
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    PIPELINE_MODE_COMPOSABLE,
)
from robot_comic.handler_factory import HandlerFactory
from robot_comic.composable_conversation_handler import ComposableConversationHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deps() -> MagicMock:
    """Return a MagicMock ToolDependencies — mirrors test_handler_factory.py."""
    return MagicMock(name="ToolDependencies")


def _build_handler(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
    llm_backend: str,
    output_backend: str,
) -> ComposableConversationHandler:
    """Build a composable handler via the factory for the given triple."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", llm_backend)

    result = HandlerFactory.build(
        AUDIO_INPUT_MOONSHINE,
        output_backend,
        mock_deps,
        pipeline_mode=PIPELINE_MODE_COMPOSABLE,
    )
    assert isinstance(result, ComposableConversationHandler)
    return result


# ---------------------------------------------------------------------------
# Bug-witness: tool_dispatcher must be wired on every tool-enabled triple.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "llm_backend,output_backend",
    [
        (LLM_BACKEND_LLAMA, AUDIO_OUTPUT_ELEVENLABS),
        (LLM_BACKEND_LLAMA, AUDIO_OUTPUT_CHATTERBOX),
        (LLM_BACKEND_GEMINI, AUDIO_OUTPUT_CHATTERBOX),
        (LLM_BACKEND_GEMINI, AUDIO_OUTPUT_ELEVENLABS),
    ],
)
def test_factory_wires_tool_dispatcher_on_composable_triples(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
    llm_backend: str,
    output_backend: str,
) -> None:
    """Every (llama|gemini) × (elevenlabs|chatterbox) composable triple must
    construct a ``ComposablePipeline`` with a non-None ``tool_dispatcher``.

    This is the §2.2 bug witness: before the fix, every factory builder
    passes no dispatcher and ``ComposablePipeline`` defaults the attribute
    to ``None``. The orchestrator's tool-call branch then breaks the loop
    without speaking on any tool-triggered turn.
    """
    handler = _build_handler(monkeypatch, mock_deps, llm_backend, output_backend)
    assert handler.pipeline.tool_dispatcher is not None, (
        f"({llm_backend}, {output_backend}) composable pipeline has no "
        "tool_dispatcher wired; LLM tool_calls will silently break the turn"
    )


# ---------------------------------------------------------------------------
# Behavior: the wired dispatcher must actually route ToolCalls through to
# the underlying tool registry (not just be a callable that does nothing).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wired_dispatcher_invokes_dispatch_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
) -> None:
    """A factory-wired dispatcher must call ``dispatch_tool_call`` with the
    ToolCall's name + JSON-serialised args and return a string result.

    Stubs the tool-layer dispatcher to capture the call, then invokes the
    pipeline's tool_dispatcher with a fake ``ToolCall``. Verifies the
    routing without dragging the real tool registry / robot deps in.
    """
    handler = _build_handler(
        monkeypatch,
        mock_deps,
        LLM_BACKEND_LLAMA,
        AUDIO_OUTPUT_ELEVENLABS,
    )

    captured: dict[str, Any] = {}

    async def _fake_dispatch(
        tool_name: str, args_json: str, deps: Any
    ) -> dict[str, Any]:
        captured["tool_name"] = tool_name
        captured["args_json"] = args_json
        captured["deps"] = deps
        return {"ok": True, "danced": "happy"}

    # The factory's dispatcher shim calls ``dispatch_tool_call`` imported
    # at module load. Patch at both the source module and the
    # ``handler_factory`` binding so whichever path the shim uses is
    # intercepted (the import-site binding is what actually fires, but
    # patching the source defends against a future refactor that calls
    # ``core_tools.dispatch_tool_call`` directly).
    import robot_comic.handler_factory as factory_mod
    import robot_comic.tools.core_tools as core_tools_mod

    monkeypatch.setattr(core_tools_mod, "dispatch_tool_call", _fake_dispatch)
    if hasattr(factory_mod, "dispatch_tool_call"):
        monkeypatch.setattr(factory_mod, "dispatch_tool_call", _fake_dispatch)

    dispatcher = handler.pipeline.tool_dispatcher
    assert dispatcher is not None  # already pinned above

    call = ToolCall(id="t-1", name="dance", args={"name": "happy"})
    result = await dispatcher(call)

    assert captured["tool_name"] == "dance"
    # The shim must serialise the args dict to JSON since the legacy
    # dispatch_tool_call signature takes ``args_json: str``.
    assert json.loads(captured["args_json"]) == {"name": "happy"}
    assert captured["deps"] is mock_deps

    # The orchestrator appends the dispatcher's return value as the
    # ``content`` of a ``role=tool`` history entry — so it must be a string.
    assert isinstance(result, str)
    # The result should round-trip the dispatched tool's result so the
    # LLM gets enough context on the next turn.
    parsed = json.loads(result)
    assert parsed == {"ok": True, "danced": "happy"}


@pytest.mark.asyncio
async def test_wired_dispatcher_returns_error_string_on_unknown_tool(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
) -> None:
    """When the tool layer returns an ``error`` payload, the dispatcher
    surfaces it as a JSON string so the LLM can see what went wrong on the
    next round-trip. Mirrors the legacy ``_dispatch_tool_call`` contract
    in ``core_tools.py:325``.
    """
    handler = _build_handler(
        monkeypatch,
        mock_deps,
        LLM_BACKEND_LLAMA,
        AUDIO_OUTPUT_ELEVENLABS,
    )

    async def _fake_dispatch(
        tool_name: str, args_json: str, deps: Any
    ) -> dict[str, Any]:
        return {"error": f"unknown tool: {tool_name}"}

    import robot_comic.handler_factory as factory_mod
    import robot_comic.tools.core_tools as core_tools_mod

    monkeypatch.setattr(core_tools_mod, "dispatch_tool_call", _fake_dispatch)
    if hasattr(factory_mod, "dispatch_tool_call"):
        monkeypatch.setattr(factory_mod, "dispatch_tool_call", _fake_dispatch)

    dispatcher = handler.pipeline.tool_dispatcher
    assert dispatcher is not None

    call = ToolCall(id="t-2", name="nonexistent_tool", args={})
    result = await dispatcher(call)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == {"error": "unknown tool: nonexistent_tool"}


# ---------------------------------------------------------------------------
# Bonus scope — `tool.execute` span emit (Rec 1 from the instrumentation
# audit memo at docs/superpowers/specs/2026-05-16-instrumentation-audit.md).
# The monitor's tool-count column reads child ``tool.execute`` spans to
# populate the Tools cell; the composable orchestrator emitted nothing
# around the dispatch site post-4d, so the column showed 0 for every
# composable turn that actually called tools.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_emits_tool_execute_span_with_outcome_success(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
) -> None:
    """A successful tool dispatch emits a ``tool.execute`` span with
    ``tool.name`` / ``tool.id`` / ``outcome=success`` attributes. The
    monitor's tool-count column depends on this span; the attribute set
    matches the legacy ``BackgroundToolManager._run_tool`` span shape so
    no monitor-side change is needed.
    """
    from opentelemetry import trace as ot_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Swap the global tracer provider for this test; restore after via
    # monkeypatch teardown. ``telemetry.get_tracer()`` calls
    # ``trace.get_tracer(...)`` which reads the global, so this captures
    # spans without re-running the full ``telemetry.init()`` machinery.
    monkeypatch.setattr(ot_trace, "_TRACER_PROVIDER", provider, raising=False)
    # Defensive: trace API has a setter too.
    try:
        ot_trace.set_tracer_provider(provider)
    except Exception:  # pragma: no cover — already set above
        pass

    handler = _build_handler(
        monkeypatch,
        mock_deps,
        LLM_BACKEND_LLAMA,
        AUDIO_OUTPUT_ELEVENLABS,
    )

    async def _fake_dispatch(
        tool_name: str, args_json: str, deps: Any
    ) -> dict[str, Any]:
        return {"ok": True}

    import robot_comic.handler_factory as factory_mod
    import robot_comic.tools.core_tools as core_tools_mod

    monkeypatch.setattr(core_tools_mod, "dispatch_tool_call", _fake_dispatch)
    if hasattr(factory_mod, "dispatch_tool_call"):
        monkeypatch.setattr(factory_mod, "dispatch_tool_call", _fake_dispatch)

    dispatcher = handler.pipeline.tool_dispatcher
    assert dispatcher is not None

    call = ToolCall(id="span-test-1", name="dance", args={"name": "happy"})
    await dispatcher(call)

    spans = [s for s in exporter.get_finished_spans() if s.name == "tool.execute"]
    assert len(spans) == 1, f"Expected one tool.execute span, got: {[s.name for s in exporter.get_finished_spans()]}"
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert attrs.get("tool.name") == "dance"
    assert attrs.get("tool.id") == "span-test-1"
    assert attrs.get("outcome") == "success"


@pytest.mark.asyncio
async def test_dispatcher_emits_tool_execute_span_with_outcome_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
) -> None:
    """A tool-layer ``error`` payload tags the span ``outcome=error`` so the
    monitor can colour the row red. The span still ends cleanly — the
    error is surfaced to the LLM as a tool-result string, not raised."""
    from opentelemetry import trace as ot_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(ot_trace, "_TRACER_PROVIDER", provider, raising=False)
    try:
        ot_trace.set_tracer_provider(provider)
    except Exception:  # pragma: no cover
        pass

    handler = _build_handler(
        monkeypatch,
        mock_deps,
        LLM_BACKEND_LLAMA,
        AUDIO_OUTPUT_ELEVENLABS,
    )

    async def _fake_dispatch(
        tool_name: str, args_json: str, deps: Any
    ) -> dict[str, Any]:
        return {"error": "boom"}

    import robot_comic.handler_factory as factory_mod
    import robot_comic.tools.core_tools as core_tools_mod

    monkeypatch.setattr(core_tools_mod, "dispatch_tool_call", _fake_dispatch)
    if hasattr(factory_mod, "dispatch_tool_call"):
        monkeypatch.setattr(factory_mod, "dispatch_tool_call", _fake_dispatch)

    dispatcher = handler.pipeline.tool_dispatcher
    assert dispatcher is not None

    call = ToolCall(id="span-test-2", name="bad_tool", args={})
    await dispatcher(call)

    spans = [s for s in exporter.get_finished_spans() if s.name == "tool.execute"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("tool.name") == "bad_tool"
    assert attrs.get("outcome") == "error"
