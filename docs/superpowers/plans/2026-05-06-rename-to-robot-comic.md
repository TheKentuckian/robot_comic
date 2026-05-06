# Rename reachy_mini_conversation_app → robot_comic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Python package, CLI entry point, app entry point key, and main class from `reachy_mini_conversation_app` / `ReachyMiniConversationApp` to `robot_comic` / `RobotComic`, preserving all existing behavior and custom persona changes.

**Architecture:** Pure rename — no logic changes. The `reachy_talk_data` sibling package and all third-party imports (`reachy_mini`, `reachy_mini_dances_library`, etc.) stay untouched. GitHub workflow repo IDs are also out of scope (they reference the git repo name, not the Python package).

**Tech Stack:** Python 3.10+, setuptools, uv, pytest, ruff

---

## File Map

| Action | Path |
|--------|------|
| Rename dir | `src/reachy_mini_conversation_app/` → `src/robot_comic/` |
| Modify | `pyproject.toml` |
| Bulk-replace imports | All `*.py` under `src/robot_comic/` (after rename) and `tests/` |
| Class rename | `src/robot_comic/main.py` line 315 |
| Title only | `README.md` frontmatter `title:` field |
| Delete (auto-regen) | `src/reachy_mini_conversation_app.egg-info/` |

---

### Task 1: Rename the package directory

**Files:**
- Rename: `src/reachy_mini_conversation_app/` → `src/robot_comic/`

- [ ] **Step 1: Rename the directory**

```powershell
Rename-Item "D:\Projects\reachy_mini_conversation_app\src\reachy_mini_conversation_app" "robot_comic"
```

Expected: No output, directory now exists as `src/robot_comic/`.

- [ ] **Step 2: Verify rename succeeded**

```powershell
Get-Item "D:\Projects\reachy_mini_conversation_app\src\robot_comic"
```

Expected: Directory listing shows `robot_comic`.

- [ ] **Step 3: Delete stale egg-info (it will regenerate on next install)**

```powershell
Remove-Item -Recurse -Force "D:\Projects\reachy_mini_conversation_app\src\reachy_mini_conversation_app.egg-info"
```

Expected: No error.

- [ ] **Step 4: Commit checkpoint**

```bash
git add -A
git commit -m "rename: move package directory reachy_mini_conversation_app → robot_comic"
```

---

### Task 2: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

Changes needed (6 locations):
1. `name = "reachy_mini_conversation_app"` → `name = "robot_comic"`
2. Script: `reachy-mini-conversation-app = "reachy_mini_conversation_app.main:main"` → `robot-comic = "robot_comic.main:main"`
3. Entry-point key: `reachy_mini_conversation_app = "reachy_mini_conversation_app.main:ReachyMiniConversationApp"` → `robot_comic = "robot_comic.main:RobotComic"`
4. Package-data key: `reachy_mini_conversation_app = [...]` → `robot_comic = [...]`
5. Ruff `known-local-folder`: `["reachy_mini_conversation_app"]` → `["robot_comic"]`

- [ ] **Step 1: Update project name (line 6)**

Edit `pyproject.toml`:
```toml
# old:
name = "reachy_mini_conversation_app"
# new:
name = "robot_comic"
```

- [ ] **Step 2: Update CLI script entry (line 75)**

```toml
# old:
reachy-mini-conversation-app = "reachy_mini_conversation_app.main:main"
# new:
robot-comic = "robot_comic.main:main"
```

- [ ] **Step 3: Update reachy_mini_apps entry point (line 86)**

```toml
# old:
reachy_mini_conversation_app = "reachy_mini_conversation_app.main:ReachyMiniConversationApp"
# new:
robot_comic = "robot_comic.main:RobotComic"
```

- [ ] **Step 4: Update package-data key (line 96)**

```toml
# old:
[tool.setuptools.package-data]
reachy_mini_conversation_app = [
# new:
[tool.setuptools.package-data]
robot_comic = [
```

- [ ] **Step 5: Update ruff known-local-folder (line 133)**

```toml
# old:
known-local-folder = ["reachy_mini_conversation_app"]
# new:
known-local-folder = ["robot_comic"]
```

- [ ] **Step 6: Verify pyproject.toml has no remaining references to old name**

```powershell
Select-String -Path "D:\Projects\reachy_mini_conversation_app\pyproject.toml" -Pattern "reachy_mini_conversation_app"
```

Expected: Zero matches.

- [ ] **Step 7: Commit checkpoint**

```bash
git add pyproject.toml
git commit -m "rename: update pyproject.toml for robot_comic package"
```

---

### Task 3: Bulk-replace all Python imports

**Files:**
- All `*.py` under `src/robot_comic/` (29 source files)
- All `*.py` under `tests/` (18 test files)
- `external_content/external_tools/starter_custom_tool.py`

The string `reachy_mini_conversation_app` appears in import paths only (not in class names, which are handled in Task 4). A global replace is safe.

- [ ] **Step 1: Replace in all Python files under src/ and tests/**

```powershell
Get-ChildItem -Recurse -Filter "*.py" -Path "D:\Projects\reachy_mini_conversation_app\src", "D:\Projects\reachy_mini_conversation_app\tests", "D:\Projects\reachy_mini_conversation_app\external_content" |
  ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match 'reachy_mini_conversation_app') {
      ($content -replace 'reachy_mini_conversation_app', 'robot_comic') | Set-Content $_.FullName -NoNewline
    }
  }
```

- [ ] **Step 2: Verify no remaining import references to old name in src/ and tests/**

```powershell
Select-String -Recurse -Pattern "reachy_mini_conversation_app" -Include "*.py" -Path "D:\Projects\reachy_mini_conversation_app\src", "D:\Projects\reachy_mini_conversation_app\tests", "D:\Projects\reachy_mini_conversation_app\external_content"
```

Expected: Zero matches.

- [ ] **Step 3: Commit checkpoint**

```bash
git add -A
git commit -m "rename: replace all reachy_mini_conversation_app imports with robot_comic"
```

---

### Task 4: Rename main app class

**Files:**
- Modify: `src/robot_comic/main.py` (line 315 and any references to `ReachyMiniConversationApp`)

- [ ] **Step 1: Find all occurrences of the old class name**

```powershell
Select-String -Recurse -Pattern "ReachyMiniConversationApp" -Include "*.py" -Path "D:\Projects\reachy_mini_conversation_app"
```

Note: Every match needs to become `RobotComic`.

- [ ] **Step 2: Replace class name in all Python files**

```powershell
Get-ChildItem -Recurse -Filter "*.py" -Path "D:\Projects\reachy_mini_conversation_app\src", "D:\Projects\reachy_mini_conversation_app\tests" |
  ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match 'ReachyMiniConversationApp') {
      ($content -replace 'ReachyMiniConversationApp', 'RobotComic') | Set-Content $_.FullName -NoNewline
    }
  }
```

- [ ] **Step 3: Verify the class definition now reads `RobotComic`**

```powershell
Select-String -Pattern "class RobotComic" -Path "D:\Projects\reachy_mini_conversation_app\src\robot_comic\main.py"
```

Expected: One match at line 315.

- [ ] **Step 4: Verify no remaining references to the old class name**

```powershell
Select-String -Recurse -Pattern "ReachyMiniConversationApp" -Include "*.py" -Path "D:\Projects\reachy_mini_conversation_app"
```

Expected: Zero matches.

- [ ] **Step 5: Commit checkpoint**

```bash
git add -A
git commit -m "rename: ReachyMiniConversationApp → RobotComic"
```

---

### Task 5: Update README frontmatter title

**Files:**
- Modify: `README.md` (frontmatter `title:` only — keep all tags including `reachy_mini_python_app`)

- [ ] **Step 1: Update title line only**

Edit `README.md` line 2:
```yaml
# old:
title: Reachy Mini Conversation App
# new:
title: Robot Comic
```

And update the H1 heading (line 14):
```markdown
# old:
# Reachy Mini conversation app
# new:
# Robot Comic
```

And update the description in short_description (line 8):
```yaml
# old:
short_description: Talk with Reachy Mini!
# new:
short_description: Robot Comic — comedic Reachy Mini voice app
```

Leave all tags unchanged, especially:
```yaml
tags:
 - reachy_mini
 - reachy_mini_python_app
```

- [ ] **Step 2: Verify tags are preserved**

```powershell
Select-String -Pattern "reachy_mini_python_app" -Path "D:\Projects\reachy_mini_conversation_app\README.md"
```

Expected: One match.

- [ ] **Step 3: Commit checkpoint**

```bash
git add README.md
git commit -m "rename: update README title to Robot Comic"
```

---

### Task 6: Verification

- [ ] **Step 1: Run reachy-mini-app-assistant check**

```bash
reachy-mini-app-assistant check .
```

Record output. Pass = no errors about missing entry points or package name mismatches.

- [ ] **Step 2: Verify Python can import the package**

```bash
python -c "import robot_comic; print('robot_comic import OK')"
```

Expected: `robot_comic import OK` (or ImportError only if optional hardware deps are absent — that's acceptable since we're not on the robot).

- [ ] **Step 3: Run the test suite**

```bash
python -m pytest tests/ -x -q 2>&1 | head -40
```

Expected: All previously-passing tests still pass. Hardware-dependent tests may skip — that's expected.

- [ ] **Step 4: Confirm no remaining stale references anywhere in the repo**

```powershell
Select-String -Recurse -Pattern "reachy_mini_conversation_app" -Include "*.py", "*.toml", "*.md", "*.txt", "*.cfg" -Path "D:\Projects\reachy_mini_conversation_app" |
  Where-Object { $_.Path -notlike "*\.git\*" -and $_.Path -notlike "*\.venv\*" -and $_.Path -notlike "*.egg-info*" -and $_.Path -notlike "*\.worktrees\*" }
```

Expected: Zero matches (or only in GitHub workflow files that reference the git repo — those are out of scope).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "rename: finalize robot_comic rename, verified with app-assistant check"
```

---

## Self-Review

**Spec coverage check:**
- [x] `pyproject.toml` project name → Task 2
- [x] `pyproject.toml` scripts entry → Task 2
- [x] `[project.entry-points."reachy_mini_apps"]` → Task 2
- [x] Package directory `src/robot_comic` → Task 1
- [x] All Python imports → Task 3
- [x] Main app class `RobotComic` → Task 4
- [x] Package data / ruff config paths → Task 2
- [x] README title (preserve `reachy_mini_python_app` tag) → Task 5
- [x] Tests/imports → Task 3
- [x] Static/asset references (none found with hardcoded old name in assets) → N/A
- [x] Verification commands → Task 6

**Out of scope (confirmed):**
- `.github/workflows/` repo IDs (reference git repo name, not Python package)
- `reachy_talk_data` (separate data package, untouched)
- `.env` files (no package name references found)
- Third-party `reachy_mini_*` imports (those are external packages, not this app)

**Placeholder scan:** No TBDs or TODOs present — all steps have exact commands.

**Type consistency:** No types involved — pure rename.
