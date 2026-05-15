**Always run `ruff check` AND `ruff format --check` from the repo root locally before pushing — NOT against specific files.**

Burned us on PR #355 (Phase 4a, 2026-05-15). I ran `ruff check src/foo.py tests/test_foo.py` locally — it passed. CI ran `ruff check` against the whole repo — failed on I001 (import sort). The I001 isort rule only triggers on whole-repo invocation, not single-file. Cost one round-trip.

Sane local pre-push checklist:

```
.venv/bin/ruff check
.venv/bin/ruff format --check
.venv/bin/mypy --pretty src/robot_comic/<new_file>.py
.venv/bin/pytest tests/ -q
```

If any step is red, do not push. Fix locally first.
