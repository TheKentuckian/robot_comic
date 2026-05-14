"""Guard against rebase-residue Config attribute drops (issue #218).

Two PRs (#212 and #216) had to hotfix the same regression: a ``-X theirs``
rebase dropped ``Config`` class attributes while their
``refresh_runtime_config_from_env()`` siblings remained, producing
``AttributeError`` at runtime when the refresh function tried to write to a
non-existent attribute.

This test parses ``config.py`` with ``ast`` and asserts that every attribute
written by ``refresh_runtime_config_from_env`` (i.e. every ``config.X = …``
assignment inside that function) is also declared in the ``Config`` class body
(i.e. appears as ``X = …`` or ``X: Type = …`` at class scope).

An optional informational note is printed when class attrs exist that are not
refreshed — those are intentional read-only-at-import attrs and are not a bug.
"""

from __future__ import annotations
import ast
import pathlib
import warnings


# Absolute path to config.py — resolved relative to this test file so the
# test works regardless of the working directory pytest is invoked from.
_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "src" / "robot_comic" / "config.py"


def _collect_config_class_attrs(tree: ast.Module) -> set[str]:
    """Return the set of attribute names declared in the ``Config`` class body.

    Includes both plain assignments (``X = expr``) and annotated assignments
    (``X: Type = expr`` / ``X: Type``).  Private names (leading underscore)
    are excluded because they are implementation-internal temporaries that are
    never intended to be refreshable config knobs.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            attrs: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            attrs.add(target.id)
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name) and not item.target.id.startswith("_"):
                        attrs.add(item.target.id)
            return attrs
    raise RuntimeError("Could not find 'Config' class in config.py — was it renamed or moved?")


def _collect_refresh_attrs(tree: ast.Module) -> set[str]:
    """Return the set of attribute names written inside ``refresh_runtime_config_from_env``.

    Collects every ``config.X = …`` assignment target inside the function,
    regardless of nesting depth (handles ``if``/``else`` branches).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "refresh_runtime_config_from_env":
            attrs: set[str] = set()
            for item in ast.walk(node):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "config"
                        ):
                            attrs.add(target.attr)
            return attrs
    raise RuntimeError("Could not find 'refresh_runtime_config_from_env' in config.py — was it renamed or removed?")


def test_refresh_attrs_are_declared_on_config_class() -> None:
    """Every attr written by refresh_runtime_config_from_env must be in Config.

    If this test fails it means a rebase (or manual edit) dropped one or more
    ``Config`` class-level attribute declarations while the corresponding
    ``config.X = …`` line in the refresh function survived.  That combination
    causes an ``AttributeError`` at runtime when the refresh fires.

    To fix: re-add the missing ``X = <default>`` line(s) to the ``Config``
    class body in ``src/robot_comic/config.py``.
    """
    source = _CONFIG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_CONFIG_PATH))

    class_attrs = _collect_config_class_attrs(tree)
    refresh_attrs = _collect_refresh_attrs(tree)

    orphans = refresh_attrs - class_attrs

    # Informational: class attrs not covered by refresh (intentional read-only
    # attrs set only at import time).  Not an error — just surfaced for
    # awareness when someone reads a test failure report.
    class_only = class_attrs - refresh_attrs
    if class_only:
        warnings.warn(
            f"Config has {len(class_only)} attr(s) not written by "
            f"refresh_runtime_config_from_env (read-only-at-import — likely intentional): "
            f"{sorted(class_only)}",
            stacklevel=1,
        )

    assert not orphans, (
        f"{len(orphans)} attribute(s) are written by refresh_runtime_config_from_env "
        f"but are NOT declared in the Config class body.  This means a rebase or "
        f"manual edit silently dropped their class-level declarations, which will "
        f"cause AttributeError at runtime.\n\n"
        f"Orphaned attribute(s): {sorted(orphans)}\n\n"
        f"Fix: add each missing attribute as a class-level assignment in the Config "
        f"class body in src/robot_comic/config.py, e.g.:\n"
        + "\n".join(f"    {attr} = <default_value>" for attr in sorted(orphans))
    )
