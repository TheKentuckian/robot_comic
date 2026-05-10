# Development Notes

## Environment setup

This project does **not** use a local `.venv`. Instead it shares the central
`/venvs/apps_venv` with `reachy_mini_conversation_app` — the two apps run
from the same interpreter on the robot, so their library versions are always
in sync at runtime.

The package is installed as an editable install inside `apps_venv`:

```
Location: /venvs/apps_venv/lib/python3.12/site-packages
Editable project location: /home/pollen/apps/robot_comic
```

This means **code changes in `src/` take effect immediately** — no reinstall
step for day-to-day development.

## uv workflow

Tell uv to target the central venv instead of creating a local one:

```bash
export UV_PROJECT_ENVIRONMENT=/venvs/apps_venv
```

Add that line to `~/.bashrc` once, then the usual commands work from the
project directory:

| Task | Command |
|---|---|
| Reinstall after editing `pyproject.toml` | `uv pip install -e .` |
| Install with extras | `uv pip install -e .[local_stt]` |
| Add a one-off package | `uv pip install some-package` |
| Update the lock file | `uv lock` |
| Lint / format / type-check | `uv pip install -e .[dev]` then call tools directly |
| Run tests | `/venvs/apps_venv/bin/python -m pytest tests/ -v` |

> **Do not use `uv sync` or `uv add`** — both prune the environment down to
> a single project's dependency set and will remove packages belonging to
> the other app sharing the venv.

Note: `CLAUDE.md` still references `uv sync` / `uv run` which assumed a
local `.venv`. Those commands need updating to match this workflow.

## Extras

| Extra | Installs | Required for |
|---|---|---|
| `local_stt` | `moonshine-voice` | Local Moonshine STT backend |
| `local_vision` | torch, transformers, accelerate | SmolVLM2 local vision |
| `yolo_vision` | ultralytics, supervision, opencv | YOLO head tracking |

`moonshine-voice` is installed in `apps_venv` and is required when
`BACKEND_PROVIDER=local_stt` (the on-robot default). The autostart launcher
(`/usr/local/bin/reachy-app-autostart.py`) runs with `/venvs/apps_venv/bin/python`,
so any extras needed at runtime must be installed into that venv.

## Shared venv and version alignment

`reachy_mini_conversation_app` (the upstream this project was forked from) is
installed from a frozen HuggingFace snapshot, not a local editable. Both apps
share `apps_venv`, so runtime library versions are always identical regardless
of what each `pyproject.toml` specifies.

When Pollen updates the upstream snapshot, check that the shared dep pins
(`gradio`, `huggingface-hub`, `fastrtc`, `openai`, `reachy-mini`) remain
compatible with the versions in this project's `pyproject.toml`.

## Running on the robot

The robot autostart flow:
1. `reachy-app-autostart.py` waits for the daemon (`/openapi.json` probe)
2. Enables motors and queues the wake-up animation
3. Execs `/venvs/apps_venv/bin/python -u -m robot_comic.main`

Daemon logs go to `/tmp/daemon.jsonl` (tmpfs — cleared on reboot).
App logs go to journald: `journalctl -u reachy-app-autostart -f`.
