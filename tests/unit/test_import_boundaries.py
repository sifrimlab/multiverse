"""Phase 0 import-boundary tests.

Asserts that the host control-plane does not pull optional scientific or
ML dependencies just by being imported.  These tests are the safety net
that must stay green throughout the Alternative-1 restructuring.

Runtime checks run in a subprocess so that other test-suite imports
cannot pollute sys.modules.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_BASE = [
    "mlflow", "optuna", "streamlit", "scanpy", "torch",
    "mudata", "anndata", "h5py", "muon", "scib_metrics", "numpy",
]

_FORBIDDEN_CONTRACT = [
    "docker", "mlflow", "optuna", "streamlit", "scanpy", "torch",
    "mudata", "anndata", "h5py", "numpy",
]


def _check_imports_clean(module: str, forbidden: list[str]) -> list[str]:
    """Run a fresh Python process and return any forbidden modules that loaded."""
    forbidden_set = "{" + ", ".join(f'"{f}"' for f in forbidden) + "}"
    code = (
        f"import sys; import {module}; "
        f"forbidden = {forbidden_set}; "
        f"loaded = sorted(forbidden & {{k.split('.')[0] for k in sys.modules}}); "
        f"print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        pytest.fail(
            f"subprocess importing {module} failed:\n{result.stderr}"
        )
    import ast
    return ast.literal_eval(result.stdout.strip() or "[]")


# ---------------------------------------------------------------------------
# Runtime boundary checks (subprocess-isolated)
# ---------------------------------------------------------------------------


def test_cli_entrypoints_does_not_load_forbidden() -> None:
    """Importing the CLI entry point must not pull in optional ML stacks."""
    loaded = _check_imports_clean("multiverse.cli_entrypoints", _FORBIDDEN_BASE)
    assert loaded == [], (
        f"multiverse.cli_entrypoints caused forbidden optional imports: {loaded}"
    )


def test_contract_does_not_load_forbidden() -> None:
    """multiverse.contract must be dependency-clean."""
    try:
        loaded = _check_imports_clean("multiverse.contract", _FORBIDDEN_CONTRACT)
    except Exception as exc:
        if "No module named" in str(exc):
            pytest.skip("multiverse.contract not yet implemented (Phase 1 pending)")
        raise
    assert loaded == [], (
        f"multiverse.contract loaded forbidden modules: {loaded}"
    )


def test_base_import_does_not_load_worker() -> None:
    """import multiverse must not trigger the worker SDK."""
    code = (
        "import sys, multiverse; "
        "leaks = [k for k in sys.modules if k.startswith('multiverse.worker')]; "
        "print(leaks)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    import ast
    leaks = ast.literal_eval(result.stdout.strip() or "[]")
    assert leaks == [], f"Base import of multiverse triggered worker load: {leaks}"


# ---------------------------------------------------------------------------
# Static grep gates — catch top-level (zero-indented) forbidden imports
# ---------------------------------------------------------------------------

# Only zero-indented "import X" or "from X import" lines are top-level.
_TOP_LEVEL_FORBIDDEN = re.compile(
    r"^(?:import|from)\s+"
    r"(?:mlflow|optuna|streamlit|scanpy|torch|mudata|anndata|h5py|muon|scib_metrics)\b",
    re.MULTILINE,
)


def test_cli_entrypoints_no_top_level_forbidden_imports() -> None:
    """Static check: cli_entrypoints.py has no top-level forbidden imports."""
    src = REPO_ROOT / "multiverse" / "cli_entrypoints.py"
    if not src.exists():
        pytest.skip("cli_entrypoints.py not found")

    text = src.read_text(encoding="utf-8")
    match = _TOP_LEVEL_FORBIDDEN.search(text)
    assert match is None, (
        f"cli_entrypoints.py has top-level forbidden import: {match.group(0)!r}"
    )


def test_evaluate_no_top_level_forbidden_imports() -> None:
    """Static check: evaluate.py has no top-level forbidden imports (Phase 7)."""
    src = REPO_ROOT / "multiverse" / "evaluate.py"
    if not src.exists():
        pytest.skip("evaluate.py not found")

    text = src.read_text(encoding="utf-8")
    match = _TOP_LEVEL_FORBIDDEN.search(text)
    assert match is None, (
        f"evaluate.py has top-level forbidden import: {match.group(0)!r}"
    )


# ---------------------------------------------------------------------------
# Phase 9 gate — the legacy worker shim and old naming are fully removed.
# This fails CI if anyone reintroduces `mvr_worker`, `sdk/mvr-worker`, or the
# old `mvexp` storage/env naming into source, tests, or container build files.
# ---------------------------------------------------------------------------

_FORBIDDEN_LEGACY_TOKENS = ("mvr_worker", "mvr-worker", "sdk/mvr-worker", "mvexp", "MVEXP")

_SCAN_ROOTS = ("multiverse", "tests", "store")
_SCAN_SUFFIXES = (".py", ".toml", ".def", ".yaml", ".yml")
_SCAN_NAMES = ("Dockerfile",)


_THIS_FILE = Path(__file__).resolve()


def _iter_scan_files():
    for root in _SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            if path.resolve() == _THIS_FILE:
                continue  # the gate names the tokens it forbids
            if path.suffix in _SCAN_SUFFIXES or path.name in _SCAN_NAMES:
                yield path


def test_no_legacy_mvr_worker_or_mvexp_references() -> None:
    """Phase 9: the worker shim and legacy ``mvexp`` naming must stay gone."""
    offenders: list[str] = []
    for path in _iter_scan_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = sorted({tok for tok in _FORBIDDEN_LEGACY_TOKENS if tok in text})
        if hits:
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel}: {hits}")
    assert not offenders, (
        "legacy mvr_worker/mvexp references reintroduced:\n  "
        + "\n  ".join(offenders)
    )
