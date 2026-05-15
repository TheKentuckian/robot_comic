# Stream A — Direct ALSA RW Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bypass the ALSA MMAP-mode signal attenuation on `reachymini_audio_src` by spawning `arecord` in RW mode from our app and feeding `LocalStream.record_loop` from it. Unblocks #314 (cold-boot STT stall).

**Architecture:** A new `audio_input` package owns an `AlsaRwCapture` class that wraps an `arecord` subprocess and exposes the same `(sample_rate, int16 ndarray)` shape that `record_loop` already consumes. `LocalStream` gains a thin `_audio_source` indirection — `DaemonAudioSource` (existing path) or `AlsaRwCapture` (new path) — picked by a new config flag `AUDIO_CAPTURE_PATH`. The daemon's TTS playback path is untouched. The daemon's MMAP capture continues to run for AEC/reference reasons; `dsnoop` supports concurrent readers.

**Tech Stack:** Python 3.12, `subprocess.Popen`, `numpy`, `pytest`, `arecord` (alsa-utils, already on Pi).

**Spec:** `docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md`

**Branch:** `fix/314-stream-a-alsa-rw-capture` (already created off `main`).

---

## File Structure

| File | Created/Modified | Responsibility |
|------|------------------|----------------|
| `src/robot_comic/audio_input/__init__.py` | Create | Package init, exports `AlsaRwCapture` |
| `src/robot_comic/audio_input/alsa_rw_capture.py` | Create | `arecord` subprocess wrapper exposing `get_audio_sample()` |
| `src/robot_comic/console.py` | Modify | Add `_AudioSource` protocol, `_DaemonAudioSource` shim, `_build_audio_source`, wire into `record_loop` + lifecycle |
| `src/robot_comic/config.py` | Modify | Add `AUDIO_CAPTURE_PATH_*` constants + `Config.AUDIO_CAPTURE_PATH` field + `refresh_runtime_config_from_env` mirror |
| `scripts/dsnoop_multireader_check.py` | Create | Standalone diagnostic — confirm RW reader works while daemon runs |
| `tests/test_alsa_rw_capture.py` | Create | Unit tests for `AlsaRwCapture` (cross-platform, no real `arecord`) |
| `tests/test_config_new_flags.py` | Modify | Add `AUDIO_CAPTURE_PATH` default + env-override tests |
| `tests/test_console_audio_source.py` | Create | Unit tests for `_build_audio_source` selection logic |

No changes to handlers, profiles, or tools. No new third-party Python deps.

---

## Task 1: Config — `AUDIO_CAPTURE_PATH` knob

**Files:**
- Modify: `src/robot_comic/config.py` (constants near other AUDIO_* env names ~L260, `Config` class field ~L940, `refresh_runtime_config_from_env` mirror ~L1197)
- Modify: `tests/test_config_new_flags.py` (append tests at end)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_new_flags.py`:

```python
import sys


def test_audio_capture_path_default_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = _reload_config(monkeypatch, {})
    assert cfg.AUDIO_CAPTURE_PATH == "alsa_rw"


def test_audio_capture_path_default_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = _reload_config(monkeypatch, {})
    assert cfg.AUDIO_CAPTURE_PATH == "daemon"


def test_audio_capture_path_explicit_daemon_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = _reload_config(monkeypatch, {"REACHY_MINI_AUDIO_CAPTURE_PATH": "daemon"})
    assert cfg.AUDIO_CAPTURE_PATH == "daemon"


def test_audio_capture_path_explicit_alsa_rw_on_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = _reload_config(monkeypatch, {"REACHY_MINI_AUDIO_CAPTURE_PATH": "alsa_rw"})
    assert cfg.AUDIO_CAPTURE_PATH == "alsa_rw"


def test_audio_capture_path_invalid_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = _reload_config(monkeypatch, {"REACHY_MINI_AUDIO_CAPTURE_PATH": "bogus"})
    assert cfg.AUDIO_CAPTURE_PATH == "alsa_rw"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_config_new_flags.py -v -k audio_capture_path
```
Expected: 5 FAILures with `AttributeError: type object 'Config' has no attribute 'AUDIO_CAPTURE_PATH'`.

- [ ] **Step 3: Add constants in `config.py`**

Insert after the `AUDIO_OUTPUT_BACKEND_ENV = ...` line (around line 266, before the `# 4th config dial: pipeline mode` comment block):

```python
# ---------------------------------------------------------------------------
# Audio capture path: bypass ALSA MMAP attenuation on reachymini_audio_src
# by spawning our own arecord subprocess in RW mode. See
# docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md.
# ---------------------------------------------------------------------------

AUDIO_CAPTURE_PATH_ENV = "REACHY_MINI_AUDIO_CAPTURE_PATH"
AUDIO_CAPTURE_PATH_DAEMON = "daemon"
AUDIO_CAPTURE_PATH_ALSA_RW = "alsa_rw"
AUDIO_CAPTURE_PATH_CHOICES: tuple[str, ...] = (
    AUDIO_CAPTURE_PATH_DAEMON,
    AUDIO_CAPTURE_PATH_ALSA_RW,
)


def _resolve_audio_capture_path() -> str:
    """Return the AUDIO_CAPTURE_PATH for this process.

    Default is alsa_rw on Linux (the only platform with arecord +
    reachymini_audio_src) and daemon elsewhere. The env var overrides either
    direction; invalid values log a warning and fall back to the platform
    default.
    """
    platform_default = (
        AUDIO_CAPTURE_PATH_ALSA_RW if sys.platform == "linux" else AUDIO_CAPTURE_PATH_DAEMON
    )
    raw = (os.getenv(AUDIO_CAPTURE_PATH_ENV) or "").strip().lower()
    if not raw:
        return platform_default
    if raw in AUDIO_CAPTURE_PATH_CHOICES:
        return raw
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Using platform default %r.",
        AUDIO_CAPTURE_PATH_ENV,
        raw,
        ", ".join(AUDIO_CAPTURE_PATH_CHOICES),
        platform_default,
    )
    return platform_default
```

Note: `logger` is defined further down (line 409) but `_resolve_audio_capture_path` is only ever called *from* the `Config` class body, which executes *after* the `logger = logging.getLogger(__name__)` line because class bodies run at import time in order. The function definition itself uses no module-level state until called.

- [ ] **Step 4: Add the field to `Config` class**

Find the line `MOONSHINE_HEARTBEAT = _env_flag("MOONSHINE_HEARTBEAT", default=False)` (around line 940) and add right below it:

```python
    AUDIO_CAPTURE_PATH = _resolve_audio_capture_path()
```

- [ ] **Step 5: Mirror into `refresh_runtime_config_from_env`**

Find the line `config.MOONSHINE_HEARTBEAT = _env_flag("MOONSHINE_HEARTBEAT", default=False)` (around line 1197) and add right below it:

```python
    config.AUDIO_CAPTURE_PATH = _resolve_audio_capture_path()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_config_new_flags.py -v -k audio_capture_path
```
Expected: 5 PASSes.

- [ ] **Step 7: Run the full config-suite to make sure nothing regressed**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_config_new_flags.py tests/test_config_attr_consistency.py tests/test_config_name_collisions.py tests/test_audio_backends_config.py -v
```
Expected: all green. The attr-consistency test scans for fields present in `Config` but missing in `refresh_runtime_config_from_env` and vice versa — both touched in steps 4 and 5, so it should stay green.

- [ ] **Step 8: Lint + types**

```bash
ruff check src/robot_comic/config.py tests/test_config_new_flags.py --fix
ruff format src/robot_comic/config.py tests/test_config_new_flags.py
mypy --pretty --show-error-codes src/robot_comic/config.py
```
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/robot_comic/config.py tests/test_config_new_flags.py
git commit -m "feat(config): add AUDIO_CAPTURE_PATH (daemon|alsa_rw)

Platform-aware default — alsa_rw on Linux, daemon elsewhere. Env var
REACHY_MINI_AUDIO_CAPTURE_PATH overrides; invalid values fall back to
the platform default with a warning. Mirrored into
refresh_runtime_config_from_env per the existing convention.

Wiring follows in subsequent commits. Refs #314."
```

---

## Task 2: `AlsaRwCapture` class

**Files:**
- Create: `src/robot_comic/audio_input/__init__.py`
- Create: `src/robot_comic/audio_input/alsa_rw_capture.py`
- Create: `tests/test_alsa_rw_capture.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alsa_rw_capture.py`:

```python
"""Unit tests for AlsaRwCapture.

These tests never spawn a real arecord process — Popen is stubbed.  This
keeps them portable to macOS/Windows CI.  The on-Pi field test covers
real-arecord behaviour and lives outside the automated suite.
"""

from __future__ import annotations

import sys
from typing import Any, List

import numpy as np
import pytest

from robot_comic.audio_input.alsa_rw_capture import AlsaRwCapture


class _FakeStdout:
    """File-like wrapper that returns scripted byte chunks then EOF."""

    def __init__(self, chunks: list[bytes]):
        self._chunks: list[bytes] = list(chunks)
        self._buffer = b""

    def read(self, n: int) -> bytes:
        # Behave like a non-blocking pipe: feed at most one scripted chunk per
        # call.  AlsaRwCapture buffers across calls to handle short reads.
        if not self._buffer and self._chunks:
            self._buffer = self._chunks.pop(0)
        if not self._buffer:
            return b""
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def close(self) -> None:
        pass


class _FakePopen:
    """Stub for subprocess.Popen — captures args and serves scripted bytes."""

    instances: list["_FakePopen"] = []

    def __init__(self, cmd: list[str], *args: Any, **kwargs: Any) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = _FakeStdout(self._scripted_chunks)
        self.stderr = _FakeStdout([])
        self.terminated = False
        self.killed = False
        self._return_code: int | None = None
        type(self).instances.append(self)

    # Class-level slot the test overrides per case.
    _scripted_chunks: list[bytes] = []

    def poll(self) -> int | None:
        return self._return_code

    def terminate(self) -> None:
        self.terminated = True
        self._return_code = 0

    def kill(self) -> None:
        self.killed = True
        self._return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        self._return_code = self._return_code if self._return_code is not None else 0
        return self._return_code


@pytest.fixture
def fake_popen(monkeypatch):
    _FakePopen.instances = []
    _FakePopen._scripted_chunks = []
    monkeypatch.setattr("robot_comic.audio_input.alsa_rw_capture.subprocess.Popen", _FakePopen)
    yield _FakePopen


def test_start_on_non_linux_raises(monkeypatch):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "darwin",
    )
    cap = AlsaRwCapture()
    with pytest.raises(RuntimeError, match="Linux-only"):
        cap.start()


def test_start_invokes_arecord_with_expected_args(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture(device="reachymini_audio_src", sample_rate=16000, channels=2)
    cap.start()
    assert len(fake_popen.instances) == 1
    cmd = fake_popen.instances[0].cmd
    assert cmd[0] == "arecord"
    assert "-D" in cmd and "reachymini_audio_src" in cmd
    assert "-r" in cmd and "16000" in cmd
    assert "-c" in cmd and "2" in cmd
    assert "-f" in cmd and "S16_LE" in cmd
    assert "-t" in cmd and "raw" in cmd
    # -M no = explicit "no MMAP" — the whole point of this module.
    assert "-M" in cmd and "no" in cmd


def test_get_audio_sample_returns_none_when_buffer_empty(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    fake_popen._scripted_chunks = []  # no bytes available
    cap = AlsaRwCapture(frame_samples=4, channels=2)
    cap.start()
    assert cap.get_audio_sample() is None


def test_get_audio_sample_returns_full_frame(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    # Build a known 4-frame stereo S16_LE buffer: 4 * 2 * 2 = 16 bytes.
    samples = np.array(
        [[100, -100], [200, -200], [300, -300], [400, -400]],
        dtype=np.int16,
    )
    fake_popen._scripted_chunks = [samples.tobytes()]
    cap = AlsaRwCapture(frame_samples=4, channels=2)
    cap.start()
    frame = cap.get_audio_sample()
    assert frame is not None
    assert frame.dtype == np.int16
    assert frame.shape == (4, 2)
    np.testing.assert_array_equal(frame, samples)


def test_get_audio_sample_handles_short_reads(monkeypatch, fake_popen):
    """Two short reads should combine into one full frame."""
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    samples = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype=np.int16,
    )
    raw = samples.tobytes()  # 16 bytes
    # Split into two chunks: 7 bytes, then 9 bytes.
    fake_popen._scripted_chunks = [raw[:7], raw[7:]]
    cap = AlsaRwCapture(frame_samples=4, channels=2)
    cap.start()
    # First call: only 7 bytes available, frame not complete.
    assert cap.get_audio_sample() is None
    # Second call: remaining 9 bytes arrive, frame completes.
    frame = cap.get_audio_sample()
    assert frame is not None
    np.testing.assert_array_equal(frame, samples)


def test_sample_rate_property_returns_configured_value(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture(sample_rate=24000)
    cap.start()
    assert cap.sample_rate == 24000


def test_stop_terminates_subprocess(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture()
    cap.start()
    cap.stop()
    assert fake_popen.instances[0].terminated is True


def test_stop_before_start_is_noop(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture()
    cap.stop()  # must not raise
    assert fake_popen.instances == []


def test_double_start_raises(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture()
    cap.start()
    with pytest.raises(RuntimeError, match="already started"):
        cap.start()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_alsa_rw_capture.py -v
```
Expected: `ModuleNotFoundError: No module named 'robot_comic.audio_input'`.

- [ ] **Step 3: Create the package init**

Create `src/robot_comic/audio_input/__init__.py`:

```python
"""Direct ALSA audio capture, bypassing the daemon's MMAP-attenuated path.

See docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md.
"""

from robot_comic.audio_input.alsa_rw_capture import AlsaRwCapture

__all__ = ["AlsaRwCapture"]
```

- [ ] **Step 4: Create the capture module**

Create `src/robot_comic/audio_input/alsa_rw_capture.py`:

```python
"""arecord-subprocess RW-mode capture from reachymini_audio_src.

Why this exists: the dsnoop device reachymini_audio_src delivers ~1/10 the
signal level under MMAP-interleaved access vs RW-interleaved access on this
hardware.  GStreamer's alsasrc defaults to MMAP with no override knob, so
the daemon's audio capture pipeline produces near-silent audio that
Moonshine correctly reports as "no speech."  By spawning arecord in RW
mode ourselves and feeding LocalStream.record_loop directly, we get the
full-level signal without touching the daemon.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlsaRwCapture:
    """Read S16_LE frames from an ALSA device in RW-interleaved mode."""

    def __init__(
        self,
        device: str = "reachymini_audio_src",
        sample_rate: int = 16000,
        channels: int = 2,
        frame_samples: int = 256,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_samples = frame_samples
        self._frame_bytes = frame_samples * channels * 2  # S16 = 2 bytes/sample
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._buf = bytearray()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        if sys.platform != "linux":
            raise RuntimeError(
                f"AlsaRwCapture is Linux-only (current platform: {sys.platform}). "
                "Use AUDIO_CAPTURE_PATH=daemon on non-Linux hosts."
            )
        if self._proc is not None:
            raise RuntimeError("AlsaRwCapture already started")
        cmd = [
            "arecord",
            "-q",
            "-D", self._device,
            "-M", "no",
            "-f", "S16_LE",
            "-r", str(self._sample_rate),
            "-c", str(self._channels),
            "-t", "raw",
        ]
        logger.info("Starting AlsaRwCapture: %s", " ".join(cmd))
        self._proc = subprocess.Popen(  # noqa: S603 (args list, no shell)
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def get_audio_sample(self) -> Optional[np.ndarray]:
        """Return next (frame_samples, channels) int16 frame, or None if no full frame ready."""
        if self._proc is None or self._proc.stdout is None:
            return None
        needed = self._frame_bytes - len(self._buf)
        if needed > 0:
            chunk = self._proc.stdout.read(needed)
            if chunk:
                self._buf.extend(chunk)
        if len(self._buf) < self._frame_bytes:
            return None
        frame_bytes = bytes(self._buf[: self._frame_bytes])
        del self._buf[: self._frame_bytes]
        return np.frombuffer(frame_bytes, dtype=np.int16).reshape(
            self._frame_samples, self._channels
        )

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.warning("arecord did not exit on SIGTERM, sending SIGKILL")
                proc.kill()
                proc.wait(timeout=1.0)
        except Exception as e:
            logger.warning("Error stopping arecord: %s", e)
        finally:
            # Drain stderr for diagnostics.
            try:
                if proc.stderr is not None:
                    err = proc.stderr.read()
                    if err:
                        logger.info("arecord stderr at shutdown: %s", err.decode(errors="replace").strip())
            except Exception:
                pass
        self._buf.clear()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_alsa_rw_capture.py -v
```
Expected: 9 PASSes.

- [ ] **Step 6: Lint + types**

```bash
ruff check src/robot_comic/audio_input/ tests/test_alsa_rw_capture.py --fix
ruff format src/robot_comic/audio_input/ tests/test_alsa_rw_capture.py
mypy --pretty --show-error-codes src/robot_comic/audio_input/
```
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/robot_comic/audio_input/ tests/test_alsa_rw_capture.py
git commit -m "feat(audio_input): AlsaRwCapture — arecord-backed RW capture

Subprocess wrapper around 'arecord -M no' that reads S16_LE frames from
reachymini_audio_src and yields (frame_samples, channels) int16 arrays
shaped identically to the daemon's get_audio_sample output. Linux-only
at start(); non-Linux raises with a clear message pointing to
AUDIO_CAPTURE_PATH=daemon.

9 unit tests cover platform gating, arecord arg construction, short-read
buffering, lifecycle (start/stop/stop-before-start/double-start), and
the sample_rate property. Tests use a stubbed Popen so they run on any
platform. Refs #314."
```

---

## Task 3: `_AudioSource` protocol + `_DaemonAudioSource` shim + `_build_audio_source` in `console.py`

This task introduces the source-abstraction interface and the daemon-side shim, without yet wiring it into `record_loop`. Doing this in its own commit keeps the diff small and the test surface clean.

**Files:**
- Modify: `src/robot_comic/console.py` (near top — add protocol + shim; in `LocalStream.__init__` — add `_audio_source` field; new method `_build_audio_source`)
- Create: `tests/test_console_audio_source.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_console_audio_source.py`:

```python
"""Unit tests for LocalStream._build_audio_source selection logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_comic.console import LocalStream, _DaemonAudioSource


def _fake_robot(sample_rate: int = 16000, frame: np.ndarray | None = None) -> MagicMock:
    robot = MagicMock()
    robot.media.get_input_audio_samplerate.return_value = sample_rate
    robot.media.get_audio_sample.return_value = frame
    return robot


def test_daemon_audio_source_delegates_to_robot_media():
    frame = np.zeros((256, 2), dtype=np.int16)
    robot = _fake_robot(sample_rate=24000, frame=frame)
    src = _DaemonAudioSource(robot)
    assert src.sample_rate == 24000
    out = src.get_audio_sample()
    assert out is frame
    robot.media.get_audio_sample.assert_called_once()


def test_daemon_audio_source_returns_none_when_robot_returns_none():
    robot = _fake_robot(frame=None)
    src = _DaemonAudioSource(robot)
    assert src.get_audio_sample() is None


def test_daemon_audio_source_start_stop_are_noops():
    robot = _fake_robot()
    src = _DaemonAudioSource(robot)
    src.start()  # must not raise
    src.stop()   # must not raise


def test_build_audio_source_returns_daemon_when_path_is_daemon(monkeypatch):
    import robot_comic.console as console_mod

    monkeypatch.setattr(console_mod.config, "AUDIO_CAPTURE_PATH", "daemon")
    robot = _fake_robot()
    stream = LocalStream.__new__(LocalStream)  # bypass __init__
    stream._robot = robot
    src = stream._build_audio_source()
    assert isinstance(src, _DaemonAudioSource)


def test_build_audio_source_returns_alsa_rw_when_path_is_alsa_rw(monkeypatch):
    import robot_comic.console as console_mod
    from robot_comic.audio_input import AlsaRwCapture

    monkeypatch.setattr(console_mod.config, "AUDIO_CAPTURE_PATH", "alsa_rw")
    robot = _fake_robot()
    stream = LocalStream.__new__(LocalStream)
    stream._robot = robot
    src = stream._build_audio_source()
    assert isinstance(src, AlsaRwCapture)


def test_build_audio_source_falls_back_to_daemon_when_robot_is_none(monkeypatch):
    """Sim mode constructs LocalStream(robot=None) for the admin UI only.

    record_loop never runs in sim mode, but _build_audio_source must
    tolerate robot=None so __init__ doesn't blow up.
    """
    import robot_comic.console as console_mod

    monkeypatch.setattr(console_mod.config, "AUDIO_CAPTURE_PATH", "daemon")
    stream = LocalStream.__new__(LocalStream)
    stream._robot = None
    src = stream._build_audio_source()
    # In sim mode we still need *something* to satisfy the type, but it
    # must not call robot.media.* (because robot is None).  None is
    # acceptable — record_loop's assert catches misuse loudly in real mode.
    assert src is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_console_audio_source.py -v
```
Expected: `ImportError: cannot import name '_DaemonAudioSource' from 'robot_comic.console'`.

- [ ] **Step 3: Add the protocol + shim near the top of `console.py`**

Find the import block that ends just before `class LocalStream` (~line 200). Add after the imports, before any class definitions:

```python
from typing import Protocol


class _AudioSource(Protocol):
    """Uniform shape for record_loop's audio producer.

    Both _DaemonAudioSource (legacy path via r.media) and AlsaRwCapture
    (direct arecord) conform to this Protocol so record_loop is agnostic.
    """

    @property
    def sample_rate(self) -> int: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_audio_sample(self) -> "np.ndarray | None": ...


class _DaemonAudioSource:
    """Adapter that exposes r.media.get_audio_sample as the _AudioSource shape."""

    def __init__(self, robot) -> None:  # type: ignore[no-untyped-def]
        self._robot = robot
        self._sample_rate = int(robot.media.get_input_audio_samplerate())

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        # Daemon's media.start_recording() is called separately in launch().
        # No additional work needed here.
        return None

    def stop(self) -> None:
        return None

    def get_audio_sample(self):  # type: ignore[no-untyped-def]
        return self._robot.media.get_audio_sample()
```

Make sure `numpy` is imported as `np` in this file (it should already be — verify with `grep '^import numpy' src/robot_comic/console.py`; add `import numpy as np` if missing).

- [ ] **Step 4: Add the `_build_audio_source` method to `LocalStream`**

Add this method to the `LocalStream` class. Place it just above `record_loop` (around line 1234):

```python
    def _build_audio_source(self) -> "_AudioSource | None":
        """Select the audio source per config.AUDIO_CAPTURE_PATH.

        Returns None when self._robot is None (sim mode) — record_loop
        never runs in that case, but __init__ may still call this to set
        the field.
        """
        if self._robot is None:
            return None
        path = getattr(config, "AUDIO_CAPTURE_PATH", "daemon")
        if path == config.AUDIO_CAPTURE_PATH_ALSA_RW:
            from robot_comic.audio_input import AlsaRwCapture
            return AlsaRwCapture()
        return _DaemonAudioSource(self._robot)
```

Verify `config` is imported at the top of `console.py` — search for `from robot_comic import config` or `import robot_comic.config as config`. If neither, use the form that already exists for accessing `config.X` constants elsewhere in the file.

Also add `self._audio_source` to `LocalStream.__init__` so it is always assignable. Find the `self._tasks: List[asyncio.Task[None]] = []` line (around line 216) and add right after it:

```python
        self._audio_source: "_AudioSource | None" = None
```

(We populate it from `launch()` in Task 4, not here, because `_build_audio_source` needs `self._robot`, which __init__ already has — but constructing the source at __init__ time would spawn the arecord subprocess before any audio pipeline is up. Delay to launch().)

- [ ] **Step 5: Run tests to verify they pass**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_console_audio_source.py -v
```
Expected: 6 PASSes.

- [ ] **Step 6: Run the existing console test to confirm no regression**

```bash
/venvs/apps_venv/bin/python -m pytest tests/test_console.py -v
```
Expected: no new failures (this test exercises construction + admin UI routes, not record_loop).

- [ ] **Step 7: Lint + types**

```bash
ruff check src/robot_comic/console.py tests/test_console_audio_source.py --fix
ruff format src/robot_comic/console.py tests/test_console_audio_source.py
mypy --pretty --show-error-codes src/robot_comic/console.py
```
Expected: clean. If mypy complains about the `np.ndarray | None` forward-ref string, replace with `numpy.ndarray | None` (no quotes) once you confirm `np` is in scope.

- [ ] **Step 8: Commit**

```bash
git add src/robot_comic/console.py tests/test_console_audio_source.py
git commit -m "refactor(console): _AudioSource protocol + _DaemonAudioSource shim

Introduces a uniform interface that wraps r.media.get_audio_sample so
record_loop can be agnostic between daemon-via-GStreamer and direct
arecord paths.  No behavioural change — record_loop still uses the
robot.media path directly; wiring follows in the next commit.

Refs #314."
```

---

## Task 4: Wire `_audio_source` into `record_loop` + lifecycle

**Files:**
- Modify: `src/robot_comic/console.py` (`launch()`, `record_loop`, `close()`)

- [ ] **Step 1: Read the current state of the three functions**

```bash
grep -n 'def launch\|def record_loop\|def close' src/robot_comic/console.py
```
Confirm line numbers match what's referenced below (`launch` ~L1000, `record_loop` ~L1234, `close` ~L1175). Adjust if the file has shifted.

- [ ] **Step 2: Update `launch()` — build + start the audio source**

Find this block (around line 1106-1110):

```python
        # Start media after key is set/available
        _robot.media.start_recording()
        _robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start
        apply_audio_startup_config(_robot, logger=logger)
```

Replace with:

```python
        # Start media after key is set/available
        _robot.media.start_recording()
        _robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start
        apply_audio_startup_config(_robot, logger=logger)

        # Build + start the audio source (daemon shim or direct ALSA RW).
        # If AUDIO_CAPTURE_PATH=alsa_rw on Linux, this spawns an arecord
        # subprocess alongside the daemon's still-running MMAP capture.
        self._audio_source = self._build_audio_source()
        if self._audio_source is not None:
            try:
                self._audio_source.start()
                logger.info(
                    "Audio source ready: %s (path=%s)",
                    type(self._audio_source).__name__,
                    config.AUDIO_CAPTURE_PATH,
                )
            except Exception as e:
                logger.error(
                    "Failed to start audio source %s: %s — falling back to daemon path",
                    type(self._audio_source).__name__,
                    e,
                )
                self._audio_source = _DaemonAudioSource(_robot)
                self._audio_source.start()
```

- [ ] **Step 3: Update `record_loop` to consume from `self._audio_source`**

Find:

```python
    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        assert self._robot is not None and self.handler is not None
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop
```

Replace with:

```python
    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        assert self._robot is not None and self.handler is not None
        assert self._audio_source is not None, (
            "record_loop requires _audio_source — launch() must have populated it"
        )
        input_sample_rate = self._audio_source.sample_rate
        logger.debug(
            "Audio recording started at %d Hz via %s",
            input_sample_rate,
            type(self._audio_source).__name__,
        )

        while not self._stop_event.is_set():
            audio_frame = self._audio_source.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop
```

- [ ] **Step 4: Update `close()` — stop the audio source before media**

Find the start of `close()` body (around line 1183):

```python
    def close(self) -> None:
        """Stop the stream and underlying media pipelines.
        ...
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            if self._robot is not None:
                self._robot.media.stop_recording()
```

Insert *before* `self._robot.media.stop_recording()`:

```python
        # Stop our audio source (terminates arecord subprocess if active).
        try:
            if self._audio_source is not None:
                self._audio_source.stop()
                self._audio_source = None
        except Exception as e:
            logger.debug(f"Error stopping audio source (may already be stopped): {e}")
```

So the final shape is:

```python
        # Stop our audio source (terminates arecord subprocess if active).
        try:
            if self._audio_source is not None:
                self._audio_source.stop()
                self._audio_source = None
        except Exception as e:
            logger.debug(f"Error stopping audio source (may already be stopped): {e}")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            if self._robot is not None:
                self._robot.media.stop_recording()
```

- [ ] **Step 5: Run the full pytest suite to check for regressions**

```bash
/venvs/apps_venv/bin/python -m pytest tests/ -v --ignore=tests/integration
```
Expected: existing tests stay green. If `tests/test_console.py` exercises `LocalStream.launch()` (it might mock `_robot`), the `_build_audio_source` call now runs — but with the mocked robot, `_DaemonAudioSource` accepts any object with `media.get_input_audio_samplerate()`. If the existing tests don't provide that, add a `MagicMock` shim in the test's fixture; do not remove the assert. If a test fails because it constructs `LocalStream` with a bare `robot=Mock()`, the fix is to set `robot.media.get_input_audio_samplerate.return_value = 16000` on that mock.

- [ ] **Step 6: Lint + types**

```bash
ruff check src/robot_comic/console.py --fix
ruff format src/robot_comic/console.py
mypy --pretty --show-error-codes src/robot_comic/console.py
```
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/robot_comic/console.py tests/
git commit -m "feat(console): wire _audio_source into record_loop + lifecycle

record_loop now reads frames from self._audio_source instead of
self._robot.media.get_audio_sample directly. launch() builds the source
based on config.AUDIO_CAPTURE_PATH and starts it (after start_recording
so the daemon's pipeline is up first when alsa_rw is selected — dsnoop
needs at least one reader active to publish frames).  close() stops the
source before the daemon's recording, so any arecord subprocess sees
SIGTERM cleanly.

On-Pi field test still required before merge. Refs #314."
```

---

## Task 5: `scripts/dsnoop_multireader_check.py` diagnostic

This is a standalone script, not part of the production code path. No automated tests — it's a developer tool that ships in the tree per the spec's "retest after daemon updates" guidance.

**Files:**
- Create: `scripts/dsnoop_multireader_check.py`

- [ ] **Step 1: Check that `scripts/` exists**

```bash
ls scripts/ 2>/dev/null | head -3
```
If `scripts/` doesn't exist, create it: `mkdir scripts` and add a `.gitkeep` if needed.

- [ ] **Step 2: Create the script**

Create `scripts/dsnoop_multireader_check.py`:

```python
#!/usr/bin/env python3
"""Sanity-check: open AlsaRwCapture while the daemon is running.

Confirms that arecord -M no on reachymini_audio_src can coexist with the
daemon's MMAP-mode GStreamer reader.  Run on the Pi as Step 1 of the
field test:

    sudo systemctl is-active reachy-mini-daemon.service
    /venvs/apps_venv/bin/python scripts/dsnoop_multireader_check.py

Prints peak/RMS over a 3-second window.  Expect peak >= ~0.3 (with the
operator clapping or speaking in the room) — full-scale signal is 1.0.
Anything <= 0.1 means the MMAP-attenuation bug has somehow propagated
to RW mode and the workaround needs revisiting.  See
docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md
and reference_alsa_mmap_attenuation (auto-memory) for context.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time

import numpy as np

from robot_comic.audio_input import AlsaRwCapture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("dsnoop_multireader_check")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="reachymini_audio_src", help="ALSA device alias")
    parser.add_argument("--duration", type=float, default=3.0, help="Capture window seconds")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate Hz")
    parser.add_argument("--channels", type=int, default=2, help="Channel count")
    args = parser.parse_args(argv)

    cap = AlsaRwCapture(
        device=args.device,
        sample_rate=args.rate,
        channels=args.channels,
        frame_samples=256,
    )
    cap.start()
    logger.info("Capturing %.1fs from %s while daemon (presumably) runs ...", args.duration, args.device)

    deadline = time.monotonic() + args.duration
    collected: list[np.ndarray] = []
    while time.monotonic() < deadline:
        frame = cap.get_audio_sample()
        if frame is None:
            time.sleep(0.005)
            continue
        collected.append(frame.copy())

    cap.stop()

    if not collected:
        logger.error("No frames received — arecord likely failed.  Check stderr above.")
        return 1

    stacked = np.concatenate(collected, axis=0).astype(np.int32)
    peak = int(np.max(np.abs(stacked)))
    rms = int(math.sqrt(float(np.mean(stacked.astype(np.int64) ** 2))))
    peak_norm = peak / 32767.0
    rms_norm = rms / 32767.0

    logger.info(
        "Frames=%d samples=%d peak=%d (%.4f) rms=%d (%.4f)",
        len(collected),
        stacked.shape[0],
        peak,
        peak_norm,
        rms,
        rms_norm,
    )

    if peak_norm < 0.15:
        logger.warning(
            "Peak %.4f is below the 0.15 threshold — this looks like the MMAP-attenuation "
            "pattern leaking into RW mode.  Re-verify with `arecord -D %s -f S16_LE -d 3 /tmp/rw.wav` "
            "and a baseline `arecord -D %s -M -f S16_LE -d 3 /tmp/mmap.wav` per the memory.",
            peak_norm,
            args.device,
            args.device,
        )
        return 2
    logger.info("PASS — full-level signal received while daemon is running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the script is executable / importable**

```bash
/venvs/apps_venv/bin/python -c "from scripts.dsnoop_multireader_check import main; print('importable')"
```
Expected: prints `importable`. (If `scripts/` isn't on `sys.path`, the script is still runnable as `python scripts/dsnoop_multireader_check.py` — the import check is just a syntax sanity gate.)

- [ ] **Step 4: Lint**

```bash
ruff check scripts/dsnoop_multireader_check.py --fix
ruff format scripts/dsnoop_multireader_check.py
```
Expected: clean. (No mypy — scripts/ is typically excluded.)

- [ ] **Step 5: Commit**

```bash
git add scripts/dsnoop_multireader_check.py
git commit -m "feat(scripts): dsnoop_multireader_check diagnostic

Standalone Pi-side sanity check: opens AlsaRwCapture against
reachymini_audio_src while the daemon is running, reports peak/RMS over
3 seconds.  Used as Step 1 of the field test for Stream A and kept in
the tree for re-running after future daemon updates per the memory note.

Refs #314."
```

---

## Task 6: Pi field test (manual gate — not part of automated execution)

This task does not produce a commit. It is the gate before opening the PR.

**Pre-flight on the Pi:**

- [ ] **Step 1: Restore the diagnostic file**

The Pi's `/home/pollen/apps/robot_comic/src/robot_comic/local_stt_realtime.py` has a `[FRAME_ENERGY]` insert at line 795 from a prior session. Backup at `/tmp/local_stt_realtime.py.bak`. From the laptop:

```bash
ssh reachy-mini-ip 'cp /tmp/local_stt_realtime.py.bak /home/pollen/apps/robot_comic/src/robot_comic/local_stt_realtime.py && grep -c FRAME_ENERGY /home/pollen/apps/robot_comic/src/robot_comic/local_stt_realtime.py'
```
Expected: `0` printed (no FRAME_ENERGY references remain).

- [ ] **Step 2: Push the branch and pull on the Pi**

```bash
git push origin fix/314-stream-a-alsa-rw-capture
ssh reachy-mini-ip 'cd /home/pollen/apps/robot_comic && git fetch origin && git checkout fix/314-stream-a-alsa-rw-capture && uv pip install -e .'
```
Expected: branch checked out, `uv pip install` reports the audio_input package picked up. If `uv pip install` errors, fall back to `/venvs/apps_venv/bin/pip install -e .`.

- [ ] **Step 3: Confirm the alias resolves**

```bash
ssh reachy-mini-ip 'arecord -L | grep -A0 -B0 reachymini_audio_src'
```
Expected: at least `reachymini_audio_src` printed.

- [ ] **Step 4: Run the diagnostic script**

Make sure the daemon is up: `ssh reachy-mini-ip 'systemctl is-active reachy-mini-daemon.service'` should print `active`. Then:

```bash
ssh reachy-mini-ip 'cd /home/pollen/apps/robot_comic && /venvs/apps_venv/bin/python scripts/dsnoop_multireader_check.py'
```

While the script counts down, clap or say "hello" near the robot (mic gain 90, AGC target 0.1 — that's the chassis state from last session). Expected: `PASS — full-level signal received while daemon is running.` with `peak >= 0.15`. If it fails, stop here — multi-reader contention is preventing the RW reader from getting signal, and the wiring step needs reconsideration.

**Live STT test:**

- [ ] **Step 5: Stop autostart and run the app foreground**

```bash
ssh reachy-mini-ip 'systemctl is-active reachy-app-autostart' # confirm starting state
ssh reachy-mini-ip 'sudo systemctl stop reachy-app-autostart 2>/dev/null ; sudo systemctl reset-failed reachy-app-autostart 2>/dev/null'
```

Then on the Pi (via a separate ssh terminal or `tmux`):

```bash
cd /home/pollen/apps/robot_comic
REACHY_MINI_AUDIO_CAPTURE_PATH=alsa_rw /venvs/apps_venv/bin/python -m robot_comic.main
```

Watch for the log line:
```
Audio source ready: AlsaRwCapture (path=alsa_rw)
```

- [ ] **Step 6: Play the test audio from the laptop**

Per the `reference_audio_playback_recipe` memory, play `hello.m4a` from the laptop while the app is running:

```powershell
$player = New-Object System.Windows.Media.MediaPlayer
$player.Open([uri]"D:\Projects\robot_comic\hello.m4a")
$player.Play()
Start-Sleep -Seconds 3
$player.Stop(); $player.Close()
```

Expected in the Pi-side app log: Moonshine logs an `on_line_completed` for the transcript "hello", llama-server responds, ElevenLabs synthesises, robot speaks.

- [ ] **Step 7: Repeat with `my name is tony.m4a`**

Same procedure with the second test file. Confirm the transcript captures "my name is tony" (approximately) and that another full turn completes.

- [ ] **Step 8: Cold-boot validation (×3)**

```bash
ssh reachy-mini-ip 'sudo systemctl enable reachy-app-autostart 2>/dev/null ; echo REACHY_MINI_AUDIO_CAPTURE_PATH=alsa_rw | sudo tee -a /home/pollen/apps/robot_comic/src/robot_comic/.env'
```

Then reboot the Pi 3 times in a row, each time waiting for autostart to come up, and play `hello.m4a` to confirm a turn completes within the first 60 seconds. Record success/failure for each boot. The PR does not merge unless 3/3 reboots succeed — that's the acceptance gate for #314.

```bash
# After each reboot:
ssh reachy-mini-ip 'sudo reboot'
# Wait ~90 seconds, then:
ssh reachy-mini-ip 'systemctl is-active reachy-app-autostart && journalctl -u reachy-app-autostart -n 50 --no-pager | grep -E "Audio source ready|on_line_completed"'
```

- [ ] **Step 9: Open the PR (only after Step 8 is 3/3 green)**

```bash
gh pr create --title "fix(#314): Stream A — direct ALSA RW capture bypasses MMAP attenuation" --body "$(cat <<'EOF'
## Summary
- Bypasses ALSA MMAP-mode signal attenuation on `reachymini_audio_src` by spawning `arecord -M no` and feeding `LocalStream.record_loop` directly.
- New `audio_input/` package with `AlsaRwCapture`; flag-gated via `REACHY_MINI_AUDIO_CAPTURE_PATH` (default `alsa_rw` on Linux, `daemon` elsewhere).
- Daemon's TTS playback path untouched; daemon's MMAP capture keeps running for AEC/reference reasons (dsnoop tolerates multi-reader).

## Test plan
- [x] Unit tests: 5 new in `test_config_new_flags.py`, 9 new in `test_alsa_rw_capture.py`, 6 new in `test_console_audio_source.py` — all green on laptop.
- [x] Lint + types clean.
- [x] On-Pi diagnostic (`scripts/dsnoop_multireader_check.py`) reports full-level signal with daemon running.
- [x] Laptop-played `hello.m4a` and `my name is tony.m4a` produce full turns through Moonshine → llama-server → ElevenLabs.
- [x] 3 consecutive cold-boot reboots each complete a turn within 60s — was previously ~1/5 hit rate.

Refs #314.
EOF
)"
```

---

## Self-Review

Spec coverage check, run after writing:

- ✅ **Capture module via subprocess arecord** → Task 2.
- ✅ **Module layout under `src/robot_comic/audio_input/`** → Task 2 file structure.
- ✅ **`AUDIO_CAPTURE_PATH` config with platform-aware default** → Task 1.
- ✅ **`_AudioSource` indirection in `console.py` with `_DaemonAudioSource` shim** → Task 3.
- ✅ **Wiring into `record_loop` + lifecycle (start in launch, stop in close)** → Task 4.
- ✅ **`scripts/dsnoop_multireader_check.py` diagnostic** → Task 5.
- ✅ **Field test gating merge (3 cold-boot reboots green)** → Task 6.

Placeholder scan: none ("TBD"/"TODO"/"add appropriate" not present).

Type consistency:
- `_AudioSource.sample_rate` is `int` (Task 3) → consumed as `int` in `record_loop` (Task 4) ✓
- `_AudioSource.get_audio_sample()` returns `np.ndarray | None` (Task 3) → matches `AlsaRwCapture` (Task 2) and `_DaemonAudioSource` (Task 3) ✓
- `_audio_source` field is `_AudioSource | None` (Task 3) → asserted non-None in `record_loop` (Task 4) ✓
- Config constants `AUDIO_CAPTURE_PATH_DAEMON` / `AUDIO_CAPTURE_PATH_ALSA_RW` (Task 1) → referenced as `config.AUDIO_CAPTURE_PATH_ALSA_RW` in `_build_audio_source` (Task 3) ✓

No gaps.
