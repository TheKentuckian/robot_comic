"""Pytest configuration for path setup."""

import os
import sys
from typing import Iterator
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
TESTS_PATH = PROJECT_ROOT / "tests"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)
os.environ["REACHY_MINI_HEAD_TRACKER"] = "off"  # disable default mediapipe in tests


# Env vars and config attributes mutated by ``console._persist_env_values`` /
# ``config.refresh_runtime_config_from_env``. Tests that POST to admin routes
# (``test_admin_pipeline_3column.py``, ``test_console.py``) trigger those
# mutation paths as a side effect, which then leak across tests within the
# same xdist worker process. Without restoration these leaks caused two
# 2026-05-16 CI flakes:
#   - ``test_run_realtime_session_passes_allocated_session_query`` (leaked
#     ``config.MODEL_NAME='gpt-realtime'`` → ``connect_kwargs["model"]`` set
#     unexpectedly).
#   - ``test_handler_factory.TestHandlerFactoryRealtimeCombinations`` for the
#     moonshine-realtime hybrids (leaked ``REACHY_MINI_LLM_BACKEND='gemini'``
#     → unsupported-combination ``NotImplementedError``).
_PERSISTED_ENV_KEYS = (
    "REACHY_MINI_PIPELINE_MODE",
    "REACHY_MINI_AUDIO_INPUT_BACKEND",
    "REACHY_MINI_AUDIO_OUTPUT_BACKEND",
    "REACHY_MINI_LLM_BACKEND",
    "MODEL_NAME",
)
_REFRESHED_CONFIG_ATTRS = (
    "PIPELINE_MODE",
    "AUDIO_INPUT_BACKEND",
    "AUDIO_OUTPUT_BACKEND",
    "LLM_BACKEND",
    "MODEL_NAME",
)


@pytest.fixture(autouse=True)
def _restore_persisted_env_and_config() -> Iterator[None]:
    """Snapshot and restore leak-prone env vars + config attrs around each test.

    Co-exists safely with ``monkeypatch``: monkeypatch's teardown runs first
    (it was registered later inside the test), restoring its own values.
    This fixture only un-does any *side-effect* mutations that escaped
    monkeypatch's tracking — see module-level comment for the offender.
    """
    env_snapshot = {key: os.environ.get(key) for key in _PERSISTED_ENV_KEYS}
    try:
        from robot_comic import config as _cfg_mod

        cfg = _cfg_mod.config
        config_snapshot = {attr: getattr(cfg, attr, None) for attr in _REFRESHED_CONFIG_ATTRS}
    except Exception:
        cfg = None
        config_snapshot = {}

    yield

    for key, value in env_snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    if cfg is not None:
        for attr, value in config_snapshot.items():
            try:
                setattr(cfg, attr, value)
            except Exception:
                pass
