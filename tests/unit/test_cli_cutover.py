"""Move-6 exit-gate tests for the CLI / GUI cutover.

Strategy v2 §6 acceptance:

    * Grep confirms GUI has no Docker client and no direct run-state
      DB writes.
    * Default CLI does not call the legacy Docker runner.

These tests do NOT spin up Docker — they exercise the import-graph and
the dispatch logic. The actual end-to-end submit-through-mvd contract is
tested in ``test_mvd_docker_executor.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. GUI has no Docker client
# ---------------------------------------------------------------------------


_GUI_FILES = sorted(
    p for p in (REPO_ROOT / "multiverse").glob("gui*.py") if p.is_file()
)
_FORBIDDEN_DOCKER = re.compile(r"^\s*(?:import|from)\s+docker\b", re.MULTILINE)
_FORBIDDEN_DOCKER_RUNNER = re.compile(
    r"^\s*(?:import|from)\s+multiverse\.runner\.docker_runner\b", re.MULTILINE
)


def test_gui_has_no_docker_client() -> None:
    assert _GUI_FILES, "expected to find GUI source files under multiverse/"
    for source in _GUI_FILES:
        text = source.read_text(encoding="utf-8")
        m = _FORBIDDEN_DOCKER.search(text)
        assert m is None, (
            f"GUI file {source.name} must not import the docker SDK; "
            f"found {m.group(0)!r}"
        )


def test_gui_has_no_legacy_docker_runner_imports() -> None:
    for source in _GUI_FILES:
        text = source.read_text(encoding="utf-8")
        m = _FORBIDDEN_DOCKER_RUNNER.search(text)
        assert m is None, (
            f"GUI file {source.name} must not import docker_runner; "
            f"found {m.group(0)!r}"
        )


# ---------------------------------------------------------------------------
# 2. GUI has no direct run-state DB writes
# ---------------------------------------------------------------------------


_DIRECT_RUN_WRITE = re.compile(
    r"(?i)(INSERT|UPDATE|DELETE)\s+(INTO\s+)?runs\b",
)


def test_gui_does_not_write_runs_table_directly() -> None:
    for source in _GUI_FILES:
        text = source.read_text(encoding="utf-8")
        m = _DIRECT_RUN_WRITE.search(text)
        assert m is None, (
            f"GUI file {source.name} must not write to the runs table "
            f"directly; found {m.group(0)!r}"
        )


# ---------------------------------------------------------------------------
# 3. Default CLI does NOT pull in the legacy docker_runner module
# ---------------------------------------------------------------------------


def test_default_cli_run_routes_through_mvd_entrypoint(monkeypatch):
    """Calling ``execute_run`` with no ``--simple`` and no ``--local``
    must dispatch to :func:`run_via_mvd`."""
    from multiverse.runner import cli as _cli

    called = {"mvd": False}

    def _fake_run_via_mvd(_args):
        called["mvd"] = True
        return 0

    monkeypatch.setattr(
        "multiverse.runner.mvd_entrypoint.run_via_mvd",
        _fake_run_via_mvd,
    )
    args = argparse.Namespace(
        simple=False,
        local=False,
        output="/tmp/_unused",
        seed=None,
        manifest=None,
    )
    with pytest.raises(SystemExit) as excinfo:
        _cli.execute_run(args)
    assert excinfo.value.code == 0
    assert called["mvd"] is True


def test_mvd_entrypoint_does_not_import_legacy_docker_runner() -> None:
    """The cutover entrypoint must not pull in ``docker_runner.py`` at
    module load time — otherwise the legacy code path is back."""
    text = (REPO_ROOT / "multiverse" / "runner" / "mvd_entrypoint.py").read_text(
        encoding="utf-8"
    )
    m = _FORBIDDEN_DOCKER_RUNNER.search(text)
    assert (
        m is None
    ), f"mvd_entrypoint must not import legacy docker_runner; got {m.group(0)!r}"


def test_legacy_flag_is_not_in_run_parser_help() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "multiverse.runner.cli", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--legacy" not in result.stdout


def test_runner_cli_does_not_import_legacy_docker_runner_at_module_load() -> None:
    text = (REPO_ROOT / "multiverse" / "runner" / "cli.py").read_text(encoding="utf-8")
    m = _FORBIDDEN_DOCKER_RUNNER.search(text)
    assert m is None, f"runner.cli must not import docker_runner; got {m.group(0)!r}"


# ---------------------------------------------------------------------------
# 4. mvd_entrypoint import-graph: clean of MLflow / Optuna / Streamlit
# ---------------------------------------------------------------------------


def test_mvd_entrypoint_does_not_eagerly_load_mlflow_optuna_streamlit() -> None:
    import subprocess

    script = (
        "import sys\n"
        "from multiverse.runner import mvd_entrypoint  # noqa\n"
        "for m in ('mlflow', 'optuna', 'streamlit'):\n"
        "    if m in sys.modules:\n"
        "        print(m)\n"
        "        raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
    ), f"mvd_entrypoint leaked: {result.stdout.strip()!r}\nstderr: {result.stderr}"


def test_mvd_entrypoint_projects_snapshot_to_rebuildable_index(tmp_path: Path) -> None:
    from multiverse.index.sqlite_index import INDEX_FILENAME, open_index
    from multiverse.mvd import PrimaryState
    from multiverse.runner.mvd_entrypoint import _project_snapshot_to_index

    _project_snapshot_to_index(
        tmp_path,
        {
            "physical_attempt_id": "attempt-cli-1",
            "logical_run_id": "logical-cli-1",
            "primary_state": PrimaryState.ARTIFACT_SUCCESS.value,
            "failure_reason": None,
            "artifact_dir": str(tmp_path / "store" / "artifacts" / "attempt-cli-1"),
            "workspace_dir": str(tmp_path / "store" / "workspaces" / "attempt-cli-1"),
            "manifest_path": "/tmp/manifest.yaml",
            "cancel_requested": False,
            "submitted_wall_iso": "2026-01-01T00:00:00+00:00",
            "options": {"dataset_slug": "demo", "model_slug": "pca"},
            "projections": {"mlflow": "TRACKING_PENDING"},
        },
    )

    with open_index(tmp_path / INDEX_FILENAME, create_if_missing=False) as index:
        row = index.get_run("attempt-cli-1")
        assert row is not None
        assert row["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        assert index.projections_for("attempt-cli-1")["mlflow"] == "TRACKING_PENDING"
