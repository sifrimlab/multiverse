"""Sole-writer invariant gate (STRATEGY G6 / M5).

Asserts that no Python file under ``multiverse/`` (outside the
designated writer modules) embeds raw SQL mutation strings. The
permitted writers are:

* ``multiverse/index/`` — kernel projection (SqliteIndex)
* ``multiverse/index_projection.py`` — read-only facade (no mutations)
* ``multiverse/asset_registry.py`` — user-managed dataset/model tables
* ``multiverse/registry_db.py`` — legacy shim (kept for compat)

Any other file that embeds ``INSERT INTO``, ``UPDATE ... SET``,
``DELETE FROM``, or ``CREATE TABLE`` is in violation of the invariant.

This test bakes the constraint into CI so a refactor cannot silently
add a new SQLite writer outside the designated modules.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MULTIVERSE_ROOT = Path(__file__).parent.parent.parent / "multiverse"

# Modules allowed to contain raw SQL mutations.
_ALLOWED_WRITER_PATTERNS = {
    "index",            # multiverse/index/ package
    "index_projection", # multiverse/index_projection.py
    "asset_registry",   # multiverse/asset_registry.py
    "registry_db",      # multiverse/registry_db.py (legacy shim)
    "models_ingest",    # multiverse/models_ingest.py — model registration (user asset)
}

# SQL mutation keywords that must not appear in disallowed files.
_SQL_MUTATION_RE = re.compile(
    r"\b(INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_allowed(rel_path: Path) -> bool:
    """Return True iff this path is one of the designated writer modules."""
    parts = rel_path.parts
    # multiverse/index/... (any file inside the index package)
    if len(parts) >= 2 and parts[0] == "multiverse" and parts[1] == "index":
        return True
    # multiverse/index_projection.py, asset_registry.py, registry_db.py
    if len(parts) == 2 and parts[0] == "multiverse":
        stem = Path(parts[1]).stem
        return stem in _ALLOWED_WRITER_PATTERNS
    return False


def _extract_string_literals(source: str) -> list[str]:
    """Extract all string literals from Python source via AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
        elif isinstance(node, (ast.JoinedStr,)):
            # f-string: collect the literal parts
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    literals.append(child.value)
    return literals


def _collect_violations() -> list[tuple[Path, str, str]]:
    """Return (file_path, sql_match, snippet) for each violation."""
    violations: list[tuple[Path, str, str]] = []
    for py_file in sorted(_MULTIVERSE_ROOT.rglob("*.py")):
        rel = py_file.relative_to(_MULTIVERSE_ROOT.parent)
        if _is_allowed(rel):
            continue
        if "__pycache__" in py_file.parts:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for literal in _extract_string_literals(source):
            m = _SQL_MUTATION_RE.search(literal)
            if m:
                snippet = literal[:120].replace("\n", " ")
                violations.append((py_file, m.group(0), snippet))
    return violations


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_no_raw_sql_mutations_outside_designated_writers() -> None:
    """No file outside the designated writer modules may embed INSERT/UPDATE/
    DELETE/CREATE TABLE SQL strings.

    If this test fails, either:
    1. Move the SQL to ``multiverse/index/sqlite_index.py`` or
       ``multiverse/asset_registry.py`` (preferred), OR
    2. Add the file to ``_ALLOWED_WRITER_PATTERNS`` with a justification.
    """
    violations = _collect_violations()
    if not violations:
        return
    lines = [
        f"  {v[0].relative_to(_MULTIVERSE_ROOT.parent)}: "
        f"found {v[1]!r} in string: {v[2]!r}"
        for v in violations
    ]
    raise AssertionError(
        f"Sole-writer invariant violated — {len(violations)} raw SQL mutation(s) "
        f"found outside designated writer modules:\n" + "\n".join(lines)
    )
