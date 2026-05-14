"""Smoke tests that expected repo-level files exist at the project root."""

from pathlib import Path


# Resolve the repo root: this file lives in tests/, one level down from root.
REPO_ROOT = Path(__file__).parent.parent


def test_now_md_exists() -> None:
    """NOW.md must exist at the repo root (issue #1)."""
    now_md = REPO_ROOT / "NOW.md"
    assert now_md.exists(), f"NOW.md not found at {now_md}"
    assert now_md.is_file(), f"{now_md} is not a regular file"


def test_now_md_has_required_sections() -> None:
    """NOW.md must contain the expected section headings."""
    content = (REPO_ROOT / "NOW.md").read_text(encoding="utf-8")
    for heading in ("## Current focus", "## In progress", "## Next up", "## Recently shipped"):
        assert heading in content, f"NOW.md is missing section: {heading!r}"


def test_contributing_md_exists() -> None:
    """CONTRIBUTING.md must exist at the repo root."""
    contrib = REPO_ROOT / "CONTRIBUTING.md"
    assert contrib.exists(), f"CONTRIBUTING.md not found at {contrib}"
    assert contrib.is_file(), f"{contrib} is not a regular file"


def test_contributing_md_has_issue_conventions() -> None:
    """CONTRIBUTING.md must contain the Issue conventions section."""
    content = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Issue conventions" in content, "CONTRIBUTING.md is missing 'Issue conventions' section"
