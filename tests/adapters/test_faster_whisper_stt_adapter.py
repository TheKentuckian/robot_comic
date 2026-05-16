"""Tests for ``FasterWhisperSTTAdapter`` — Phase 5f.

The adapter wraps faster-whisper + silero-vad. Neither dependency is
installed in CI; tests stub both via ``sys.modules`` injection so the
adapter's import-on-start path picks up the stubs. The stubs simulate:

- ``faster_whisper.WhisperModel(...)`` returning an object whose
  ``transcribe(audio, language=...)`` method returns a ``(segments, info)``
  pair. ``segments`` is an iterable of objects exposing ``.text``.
- ``silero_vad.load_silero_vad()`` returning an opaque "model" handle.
- ``silero_vad.VADIterator(model, sampling_rate=...)`` returning a callable
  that drives the chunker — feed it a 512-sample float32 chunk, get back
  ``None`` / ``{"start": idx}`` / ``{"end": idx}``.
"""

from __future__ import annotations
import sys
import types
from typing import Any

import numpy as np
import pytest

from robot_comic.backends import AudioFrame, STTBackend


# ---------------------------------------------------------------------------
# Stub injection helpers
# ---------------------------------------------------------------------------


class _StubSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubWhisperModel:
    """Stand-in for ``faster_whisper.WhisperModel``.

    Records constructor args + ``transcribe`` calls; returns a queued list
    of segments for each call so tests can stage multi-utterance scenarios.
    """

    def __init__(self, model_name: str, *, device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[np.ndarray] = []
        # Tests push transcript strings here; each transcribe call pops the
        # next entry (or returns "" if empty).
        self.queued_results: list[str] = []
        self.close_called = False

    def transcribe(self, audio: np.ndarray, language: str = "en") -> tuple[Any, dict[str, Any]]:
        self.transcribe_calls.append(audio)
        text = self.queued_results.pop(0) if self.queued_results else ""
        if isinstance(text, list):
            segments = [_StubSegment(t) for t in text]
        else:
            segments = [_StubSegment(text)] if text else []
        return iter(segments), {"language": language}

    def close(self) -> None:
        self.close_called = True


class _StubVADIterator:
    """Stand-in for ``silero_vad.VADIterator``.

    Tests configure ``script`` — a list of events the iterator returns in
    order, one per ``__call__`` invocation. After the script runs out,
    returns ``None``. The iterator records every chunk it received.
    """

    def __init__(self, model: Any, sampling_rate: int = 16000) -> None:
        self.model = model
        self.sampling_rate = sampling_rate
        self.script: list[Any] = []
        self.received_chunks: list[np.ndarray] = []
        self.reset_called = False

    def __call__(self, chunk: np.ndarray) -> Any:
        self.received_chunks.append(chunk)
        if not self.script:
            return None
        return self.script.pop(0)

    def reset_states(self) -> None:
        self.reset_called = True


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[_StubWhisperModel, _StubVADIterator]:
    """Inject the stub modules into ``sys.modules`` and return the instances.

    The adapter's ``_load_model_and_vad`` imports ``faster_whisper`` and
    ``silero_vad`` inside a function body. Replacing the module in
    ``sys.modules`` makes the next ``from … import …`` resolve to our stub.

    Returns the (model, vad_iterator) instances the stubbed factories
    produce so tests can poke them.
    """
    # Build container-objects so the factories can stash the constructed
    # instance for the test to read.
    holder: dict[str, Any] = {"model": None, "vad_iter": None, "vad_model": object()}

    def _make_model(name: str, *, device: str = "cpu", compute_type: str = "int8") -> _StubWhisperModel:
        m = _StubWhisperModel(name, device=device, compute_type=compute_type)
        holder["model"] = m
        return m

    def _load_silero_vad() -> Any:
        return holder["vad_model"]

    def _make_vad_iterator(model: Any, sampling_rate: int = 16000) -> _StubVADIterator:
        it = _StubVADIterator(model, sampling_rate=sampling_rate)
        holder["vad_iter"] = it
        return it

    fw_module = types.ModuleType("faster_whisper")
    fw_module.WhisperModel = _make_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    sv_module = types.ModuleType("silero_vad")
    sv_module.load_silero_vad = _load_silero_vad  # type: ignore[attr-defined]
    sv_module.VADIterator = _make_vad_iterator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", sv_module)

    # Return placeholders — the real instances aren't created until start()
    # runs. Tests should grab them via the holder after start().
    return holder  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Construction + Protocol conformance
# ---------------------------------------------------------------------------


def test_constructor_accepts_no_arguments() -> None:
    """Standalone constructor (no host handler) is the only shape."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    assert adapter._model is None
    assert adapter._vad_iterator is None


def test_constructor_accepts_should_drop_frame() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter(should_drop_frame=lambda: True)
    assert adapter._should_drop_frame is not None
    assert adapter._should_drop_frame() is True


def test_constructor_accepts_model_overrides() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter(model_name="small.en", compute_type="float16")
    assert adapter._model_name == "small.en"
    assert adapter._compute_type == "float16"


def test_adapter_satisfies_stt_backend_protocol() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    assert isinstance(adapter, STTBackend)


@pytest.mark.asyncio
async def test_reset_per_session_state_is_noop() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    await adapter.reset_per_session_state()


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_loads_model_and_vad_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)

    assert holder["model"] is not None
    assert holder["model"].model_name == "tiny.en"
    assert holder["model"].compute_type == "int8"
    assert holder["vad_iter"] is not None
    assert holder["vad_iter"].sampling_rate == 16000


@pytest.mark.asyncio
async def test_start_passes_through_model_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(model_name="base.en", compute_type="int8_float32")

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert holder["model"].model_name == "base.en"
    assert holder["model"].compute_type == "int8_float32"


@pytest.mark.asyncio
async def test_start_records_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None: ...

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)
    assert adapter._on_completed is _on_completed
    assert adapter._on_speech_started is _on_speech_started


@pytest.mark.asyncio
async def test_start_accepts_on_partial_but_does_not_store_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """faster-whisper is batch; ``on_partial`` is documented as never fired."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    async def _on_partial(_t: str) -> None: ...

    await adapter.start(_on_completed, on_partial=_on_partial)
    # The adapter logs a debug message but does not store the partial cb;
    # the simplest behavioural assertion is that an attempt to fire a
    # partial-style event below produces no callback invocation. (Covered
    # by the missing-import-attribute spot-check.)
    assert not hasattr(adapter, "_on_partial")


@pytest.mark.asyncio
async def test_start_idempotent_does_not_reload_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    model_after_first = holder["model"]
    await adapter.start(_cb)
    # The stub factory replaces holder["model"] on EVERY call; a re-load
    # would have produced a different instance.
    assert holder["model"] is model_after_first


@pytest.mark.asyncio
async def test_start_load_failure_clears_state_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model load raises, the adapter resets so a retry is possible."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    # Stub both modules so the import succeeds — the failure has to come
    # from the model constructor itself, not a missing-dep ImportError.
    _install_stubs(monkeypatch)

    fw_module = types.ModuleType("faster_whisper")

    def _bad_model(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("model load boom")

    fw_module.WhisperModel = _bad_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    with pytest.raises(RuntimeError, match="model load boom"):
        await adapter.start(_cb)

    assert adapter._model is None
    assert adapter._vad_iterator is None
    assert adapter._on_completed is None


@pytest.mark.asyncio
async def test_start_raises_dependency_error_when_faster_whisper_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters.faster_whisper_stt_adapter import (
        FasterWhisperSTTAdapter,
        FasterWhisperSTTDependencyError,
    )

    # Force the import inside _load_model_and_vad to fail by injecting a
    # module that raises on attribute access.
    fw_module = types.ModuleType("faster_whisper")
    # Don't set WhisperModel — `from faster_whisper import WhisperModel` will
    # raise ImportError.
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    with pytest.raises(FasterWhisperSTTDependencyError):
        await adapter.start(_cb)


# ---------------------------------------------------------------------------
# feed_audio → VAD chunking → on_speech_started
# ---------------------------------------------------------------------------


def _silence_frame(num_samples: int = 1024, sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(samples=np.zeros(num_samples, dtype=np.int16), sample_rate=sample_rate)


@pytest.mark.asyncio
async def test_feed_audio_chunks_into_512_sample_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1024-sample frame → 2 chunks of 512 samples each into the VAD iterator."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=1024))

    vad = holder["vad_iter"]
    assert len(vad.received_chunks) == 2
    assert all(c.shape == (512,) for c in vad.received_chunks)


@pytest.mark.asyncio
async def test_feed_audio_starts_utterance_on_vad_start_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)

    # First chunk fires "start", second is silent — no transcribe yet.
    holder["vad_iter"].script = [{"start": 0}, None]
    await adapter.feed_audio(_silence_frame(num_samples=1024))

    assert started_calls == 1
    assert holder["model"].transcribe_calls == []


@pytest.mark.asyncio
async def test_feed_audio_does_not_call_speech_started_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``on_speech_started`` is fine — VAD start events are silently dropped."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    await adapter.start(_on_completed)
    holder["vad_iter"].script = [{"start": 0}]
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=512))


# ---------------------------------------------------------------------------
# VAD end event → transcribe → on_completed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_transcribes_and_fires_on_completed_on_vad_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)

    # Chunk 1: start. Chunks 2-3: in-speech. Chunk 4: end.
    holder["vad_iter"].script = [{"start": 0}, None, None, {"end": 0}]
    holder["model"].queued_results = ["hello robot"]

    await adapter.feed_audio(_silence_frame(num_samples=512 * 4))

    assert captured == ["hello robot"]
    assert len(holder["model"].transcribe_calls) == 1


@pytest.mark.asyncio
async def test_feed_audio_joins_multi_segment_transcribe_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    holder["vad_iter"].script = [{"start": 0}, {"end": 0}]
    holder["model"].queued_results = [["hello", " robot", " how are you"]]
    await adapter.feed_audio(_silence_frame(num_samples=512 * 2))

    assert captured == ["hello robot how are you"]


@pytest.mark.asyncio
async def test_feed_audio_empty_transcribe_result_does_not_fire_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    holder["vad_iter"].script = [{"start": 0}, {"end": 0}]
    holder["model"].queued_results = [""]
    await adapter.feed_audio(_silence_frame(num_samples=512 * 2))

    assert captured == []


@pytest.mark.asyncio
async def test_feed_audio_whitespace_transcribe_result_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    holder["vad_iter"].script = [{"start": 0}, {"end": 0}]
    holder["model"].queued_results = ["   "]
    await adapter.feed_audio(_silence_frame(num_samples=512 * 2))

    assert captured == []


@pytest.mark.asyncio
async def test_feed_audio_back_to_back_utterances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two utterances in one stream → two on_completed calls."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    holder["vad_iter"].script = [
        {"start": 0},  # utterance 1 starts
        {"end": 0},  # utterance 1 ends
        {"start": 0},  # utterance 2 starts
        {"end": 0},  # utterance 2 ends
    ]
    holder["model"].queued_results = ["first", "second"]
    await adapter.feed_audio(_silence_frame(num_samples=512 * 4))

    assert captured == ["first", "second"]


# ---------------------------------------------------------------------------
# should_drop_frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_should_drop_frame_when_true_skips_feed_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(should_drop_frame=lambda: True)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=512))

    assert holder["vad_iter"].received_chunks == []
    assert holder["model"].transcribe_calls == []


@pytest.mark.asyncio
async def test_should_drop_frame_when_false_processes_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(should_drop_frame=lambda: False)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=512))

    assert len(holder["vad_iter"].received_chunks) == 1


@pytest.mark.asyncio
async def test_should_drop_frame_default_none_processes_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()  # no should_drop_frame

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=1024))

    assert len(holder["vad_iter"].received_chunks) == 2


# ---------------------------------------------------------------------------
# feed_audio guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_before_start_is_safe_noop() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    # Must not raise — silently drops the frame because model/VAD aren't loaded.
    await adapter.feed_audio(_silence_frame(num_samples=512))


@pytest.mark.asyncio
async def test_feed_audio_handles_list_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AudioFrame.samples`` may be a Python list, not always ndarray."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(AudioFrame(samples=[0] * 512, sample_rate=16000))

    assert len(holder["vad_iter"].received_chunks) == 1


@pytest.mark.asyncio
async def test_feed_audio_resamples_when_sample_rate_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-16kHz frames are resampled before chunking."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    # 1024 samples at 24 kHz → ~683 samples at 16 kHz → 1 full 512-sample
    # chunk plus a remainder. Assert at least one chunk made it through.
    await adapter.feed_audio(AudioFrame(samples=np.zeros(1024, dtype=np.int16), sample_rate=24000))
    assert len(holder["vad_iter"].received_chunks) >= 1


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_clears_state_and_releases_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()

    assert adapter._model is None
    assert adapter._vad_iterator is None
    assert adapter._on_completed is None
    assert holder["model"].close_called is True
    assert holder["vad_iter"].reset_called is True


@pytest.mark.asyncio
async def test_stop_is_safe_when_never_started() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    await adapter.stop()  # must not raise
    assert adapter._model is None


@pytest.mark.asyncio
async def test_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()
    await adapter.stop()  # second call must not raise


# ---------------------------------------------------------------------------
# Callback exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_completed_exception_does_not_crash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _bad(_t: str) -> None:
        raise ValueError("user code bug")

    await adapter.start(_bad)
    holder["vad_iter"].script = [{"start": 0}, {"end": 0}]
    holder["model"].queued_results = ["payload"]
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=512 * 2))


@pytest.mark.asyncio
async def test_on_speech_started_exception_does_not_crash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    async def _bad_started() -> None:
        raise ValueError("user code bug")

    await adapter.start(_cb, on_speech_started=_bad_started)
    holder["vad_iter"].script = [{"start": 0}]
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=512))


@pytest.mark.asyncio
async def test_transcribe_exception_does_not_crash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transcribe-call failure logs but doesn't kill the adapter."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)

    def _bad_transcribe(_a: Any, language: str = "en") -> Any:
        raise RuntimeError("transcribe boom")

    holder["model"].transcribe = _bad_transcribe  # type: ignore[method-assign]
    holder["vad_iter"].script = [{"start": 0}, {"end": 0}]
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=512 * 2))
    assert captured == []
