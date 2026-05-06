# Robot Comic install plan

## Understanding

This repo is a public fork of the Reachy Mini conversation app that should be installed as a separate Reachy Mini app named `robot_comic` / `Robot Comic`. It must not overwrite or uninstall the official conversation app already installed on this Wireless Reachy Mini.

Desired identifiers:

- Python package/module: `robot_comic`
- Reachy Mini app entry point: `robot_comic`
- App class: `RobotComic`
- Display name: `Robot Comic`
- Runtime environment: `/venvs/apps_venv/`

## Technical approach

1. Use the existing clone at `/home/pollen/apps/robot_comic`.
2. Prefer `uv` now that it is installed in `/venvs/apps_venv/bin/uv`; fall back to `/venvs/apps_venv/bin/python -m pip` if needed.
3. Verify and, where needed, fix packaging metadata, entry points, imports, package data, CLI naming, and app class naming so this remains standalone from the official conversation app.
4. Preserve the `reachy_mini_python_app` README tag and existing persona/custom behavior.
5. Install editable into `/venvs/apps_venv/`.
6. Validate with `reachy-mini-app-assistant check .` and entry point inspection.
7. Restart `reachy-mini-daemon` only if installation and checks succeed.

## Questions or blockers

No blocking questions. `/home/pollen/apps/robot_comic` is already cloned and will be used as the app repo.
