"""``multiverse slurm-submit`` CLI verb (STRATEGY M4).

The verb wires :class:`MvdSlurmExecutor` to the top-level command so
an HPC user can dispatch a single attempt without going through the
legacy planner. We exercise the happy path against the in-memory
Slurm engine by monkey-patching ``RealSlurmEngine`` at the
import site.
"""

from __future__ import annotations

import json
import threading
import time
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pytest

from multiverse import cli_entrypoints
from multiverse.slurm import InMemorySlurmEngine, SlurmJobState

pytestmark = pytest.mark.control_plane


class _DrivingSlurmEngine(InMemorySlurmEngine):
    """In-memory engine that auto-completes any submitted job after
    dropping a synthetic embeddings.h5 into the workspace."""

    def __init__(self) -> None:
        super().__init__()

    def submit(self, spec, *, script_dir):  # type: ignore[override]
        sub = super().submit(spec, script_dir=script_dir)
        workspace = spec.workspace

        def _drive() -> None:
            time.sleep(0.01)
            self.simulate_running(sub.job_id)
            with h5py.File(workspace / "embeddings.h5", "w") as f:
                f.create_dataset("latent", data=np.zeros((4, 4), dtype=np.float32))
            self.simulate_completed(sub.job_id, exit_code=0)

        threading.Thread(target=_drive, daemon=True).start()
        return sub


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> _DrivingSlurmEngine:
    engine = _DrivingSlurmEngine()

    # The slurm-submit CLI imports RealSlurmEngine inside the verb to
    # keep module-load cheap; patch the *source* attribute so the
    # import resolves to our fake.
    import multiverse.slurm.engine as engine_module

    class _Factory:
        def __init__(self, *_, **__) -> None:
            pass

        def __new__(cls, *_, **__):  # type: ignore[override]
            return engine

    monkeypatch.setattr(engine_module, "RealSlurmEngine", _Factory)
    import multiverse.slurm as slurm_pkg

    monkeypatch.setattr(slurm_pkg, "RealSlurmEngine", _Factory)
    return engine


def _capture(args: list[str]) -> tuple[int, str, str]:
    import sys

    out, err = StringIO(), StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = cli_entrypoints.slurm_submit_main(args)
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
    return rc, out.getvalue(), err.getvalue()


def test_slurm_submit_happy_path_returns_zero(
    tmp_path: Path, fake_engine: _DrivingSlurmEngine
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    sif = tmp_path / "model.sif"
    sif.write_bytes(b"sif-bytes")

    rc, stdout, stderr = _capture(
        [
            "--state-root",
            str(state_root),
            "--model-slug",
            "pca",
            "--image-sif",
            str(sif),
            "--image-digest",
            "sha256:" + "a" * 64,
            "--dataset-slug",
            "demo",
            "--dataset-path",
            str(dataset),
            "--dataset-n-obs",
            "4",
            "--dataset-n-vars",
            "8",
            "--params-json",
            '{"n_components": 4}',
            "--max-inflight",
            "4",
        ]
    )
    assert rc == 0, stderr
    payload = json.loads(stdout)
    assert payload["primary_state"] == "ARTIFACT_SUCCESS", payload
    # The fake engine recorded a single dispatch.
    assert fake_engine.submit_count == 1


def test_slurm_submit_rejects_invalid_params_json(tmp_path: Path) -> None:
    rc, stdout, stderr = _capture(
        [
            "--state-root",
            str(tmp_path / "state"),
            "--model-slug",
            "pca",
            "--image-sif",
            str(tmp_path / "x.sif"),
            "--dataset-slug",
            "demo",
            "--dataset-path",
            str(tmp_path / "data.h5mu"),
            "--dataset-n-obs",
            "4",
            "--params-json",
            "not-json",
        ]
    )
    assert rc == 2
    assert "not valid JSON" in stderr


def test_slurm_submit_rejects_params_not_an_object(tmp_path: Path) -> None:
    rc, _, stderr = _capture(
        [
            "--state-root",
            str(tmp_path / "state"),
            "--model-slug",
            "pca",
            "--image-sif",
            str(tmp_path / "x.sif"),
            "--dataset-slug",
            "demo",
            "--dataset-path",
            str(tmp_path / "data.h5mu"),
            "--dataset-n-obs",
            "4",
            "--params-json",
            "[1, 2]",
        ]
    )
    assert rc == 2
    assert "JSON object" in stderr


def test_slurm_submit_registered_in_commands_table() -> None:
    assert "slurm-submit" in cli_entrypoints.COMMANDS
    assert cli_entrypoints.COMMANDS["slurm-submit"] is cli_entrypoints.slurm_submit_main
