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

## Development workflow

When working on this repo, **push incremental changes to your branch early and often**—don't wait to finish everything locally before validating. GitHub Actions will automatically run:

- `pytest` — unit tests across Linux, macOS, and Windows
- `ruff check` — linting
- `ruff format` — code formatting
- `mypy` — type checking

**Workflow:**
1. Create a feature branch: `feat/<issue>-<desc>` or `fix/<issue>-<desc>`
2. Commit and push your changes
3. Check the workflow results in the branch's commit checks or PR
4. Address any failures and push fixes
5. Once CI passes, open a PR or continue development

This keeps iteration cycles tight and catches platform-specific or dependency issues early.

## Running on the robot

The robot autostart flow:
1. `wait-for-reachy-daemon.sh` polls `http://127.0.0.1:8000/api/motors/get_mode`
   every 200 ms until the daemon responds (up to 10 s).  Exits 1 on timeout so
   systemd can retry rather than proceed with motors disabled.
   Previously: `ExecStartPre=/bin/sleep 30` (wasted ~25 s on every boot).
2. Enables motors via `curl … motors/set_mode/enabled`
3. Queues the wake-up animation via `curl … move/play/wake_up`
4. Execs `/venvs/apps_venv/bin/python -u -m robot_comic.main`

The systemd unit and poll script live in this repo:
- `scripts/wait-for-reachy-daemon.sh` — copy to `/usr/local/bin/` (chmod +x)
- `deploy/systemd/reachy-app-autostart.service` — copy to `/etc/systemd/system/`

See the header comment in `deploy/systemd/reachy-app-autostart.service` for
one-liner install instructions.

Daemon logs go to `/tmp/daemon.jsonl` (tmpfs — cleared on reboot).
App logs go to journald: `journalctl -u reachy-app-autostart -f`.

### Log retention

On first setup, run:

```bash
sudo ./scripts/install-pi-journald.sh
```

This caps journald at 500 MB and 2-week retention so the SD card doesn't fill
up.  It also creates `/var/log/journal/` so logs survive reboots (without that
directory, journald writes to the RAM-backed `/run/log/journal` and all history
is lost on every reboot).
