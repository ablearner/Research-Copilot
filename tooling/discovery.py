"""Auto-discovery of self-registering tool modules.

Scans Python files under a tools directory for top-level
``registry.register()`` calls and imports them to trigger registration.
"""

from __future__ import annotations

import ast
import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _module_registers_tools(path: Path) -> bool:
    """Return True if *path* contains a top-level ``registry.register()`` call."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(getattr(stmt.value.func, "attr", None), str)
        and stmt.value.func.attr == "register"
        for stmt in tree.body
    )


def discover_tools(tools_dir: Path | None = None) -> list[str]:
    """Auto-import tool modules that self-register via ``registry.register()``.

    Returns a list of successfully imported module names.
    """
    tools_path = tools_dir or Path(__file__).parent.parent / "tools"
    if not tools_path.is_dir():
        logger.debug("Tools directory does not exist: %s", tools_path)
        return []

    imported: list[str] = []
    for path in sorted(tools_path.glob("**/*.py")):
        if path.name.startswith("_"):
            continue
        if not _module_registers_tools(path):
            continue
        try:
            rel = path.relative_to(tools_path.parent)
        except ValueError:
            continue
        module_name = str(rel).replace("/", ".").removesuffix(".py")
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
            logger.debug("Auto-imported tool module: %s", module_name)
        except Exception:
            logger.warning("Failed to auto-import tool module: %s", module_name, exc_info=True)
    return imported
