"""Tests for the local (no-Docker) model runner."""
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SDK = ROOT / "sdk" / "mvr-worker"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))


def _make_job(dataset_path: str, output_path: str, model_slug: str = "pca") -> dict:
    return {
        "dataset_id": 1,
        "dataset_name": "test-ds",
        "dataset_path": dataset_path,
        "model_slug": model_slug,
        "model_name_orig": model_slug,
        "model_version": "1.0.0",
        "model_params": {},
        "batch_key": "batch",
        "cell_type_key": "cell_type",
        "output_path": output_path,
    }


@pytest.fixture()
def dummy_h5mu(tmp_path):
    path = tmp_path / "dummy.h5mu"
    path.write_bytes(b"stub")
    return path


@pytest.fixture()
def isolated_registry(tmp_path, monkeypatch):
    from multiverse import registry_db

    store_dir = tmp_path / "store"
    monkeypatch.setattr(registry_db, "DB_NAME", str(tmp_path / "state.db"))
    monkeypatch.setattr(registry_db, "STORE_DIR", str(store_dir))
    monkeypatch.setattr(registry_db, "DATASETS_DIR", str(store_dir / "datasets"))
    monkeypatch.setattr(registry_db, "RAW_DATASETS_DIR", str(store_dir / "datasets" / "raw"))
    monkeypatch.setattr(registry_db, "MODELS_DIR", str(store_dir / "models"))
    monkeypatch.setattr(registry_db, "ARTIFACTS_DIR", str(store_dir / "artifacts"))
    monkeypatch.setattr(registry_db, "WORKSPACES_DIR", str(store_dir / "workspaces"))
    registry_db.init_db()
    return registry_db


def test_model_entrypoint_found(tmp_path, monkeypatch):
    """Should resolve to container/run.py under MODELS_DIR."""
    import multiverse.runner.local_runner as lr

    fake_models_dir = tmp_path / "models"
    run_py = fake_models_dir / "pca" / "container" / "run.py"
    run_py.parent.mkdir(parents=True)
    run_py.write_text("# stub", encoding="utf-8")

    monkeypatch.setattr(lr, "MODELS_DIR", str(fake_models_dir))
    result = lr._model_entrypoint("pca")
    assert result == run_py


def test_model_entrypoint_missing(tmp_path, monkeypatch):
    """Should raise FileNotFoundError when container/run.py does not exist."""
    import multiverse.runner.local_runner as lr

    monkeypatch.setattr(lr, "MODELS_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="pca"):
        lr._model_entrypoint("pca")


def test_write_job_spec(tmp_path):
    """job_spec.json must contain seed, hyperparameters, and identifiers."""
    import multiverse.runner.local_runner as lr

    job = _make_job("/data/ds.h5mu", str(tmp_path / "out"), "pca")
    job["_local_run_id"] = "run_abc123"
    path = lr._write_job_spec(tmp_path, job, seed=7)

    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["seed"] == 7
    assert spec["run_id"] == "run_abc123"
    assert spec["model_name"] == "pca"
    assert isinstance(spec["hyperparameters"], dict)


@pytest.fixture()
def fake_models_dir(tmp_path, monkeypatch, isolated_registry):
    """Create a minimal stub container/run.py under a fake MODELS_DIR."""
    import multiverse.runner.local_runner as lr

    models_dir = tmp_path / "models"
    run_py = models_dir / "pca" / "container" / "run.py"
    run_py.parent.mkdir(parents=True)
    run_py.write_text(
        textwrap.dedent(
            """            import json, os
            out = os.environ["MVR_OUTPUT_DIR"]
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"score": 0.9, "history": [0.1, 0.9]}, f)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lr, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(lr, "WORKSPACES_DIR", str(tmp_path / "workspaces"))
    monkeypatch.setattr(lr, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    return models_dir


def test_run_model_local_success(tmp_path, fake_models_dir, dummy_h5mu, isolated_registry):
    """run_model_local should persist status, metrics, logs, and copied artifacts."""
    import multiverse.runner.local_runner as lr

    output_path = tmp_path / "output"
    job = _make_job(str(dummy_h5mu), str(output_path))
    statuses: list[tuple[str, str]] = []

    result = asyncio.run(lr.run_model_local(job, seed=42, status_callback=lambda n, s: statuses.append((n, s))))

    assert result == "success"
    assert output_path.exists()
    assert (output_path / "metrics.json").exists()
    assert (output_path / "container.log").exists()
    assert any(s == "Success" for _, s in statuses)

    conn = isolated_registry.get_db_connection()
    try:
        run = conn.execute(
            "SELECT run_id, status, output_path, failure_reason FROM runs WHERE dataset_id = ?",
            (job["dataset_id"],),
        ).fetchone()
        assert run is not None
        run_id, status, stored_output, failure_reason = run
        assert status == "SUCCESS"
        assert stored_output == str(output_path)
        assert failure_reason is None

        metric_rows = conn.execute(
            "SELECT metric_name, metric_value, metric_kind FROM run_metrics WHERE run_id = ? ORDER BY metric_name",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    assert ("history", 0.9, "history_summary") in metric_rows
    assert ("score", 0.9, "scalar") in metric_rows


def test_run_model_local_script_failure(tmp_path, monkeypatch, dummy_h5mu, isolated_registry):
    """run_model_local should return 'failed' and record a failed run on non-zero exit."""
    import multiverse.runner.local_runner as lr

    models_dir = tmp_path / "models"
    run_py = models_dir / "pca" / "container" / "run.py"
    run_py.parent.mkdir(parents=True)
    run_py.write_text("import sys; sys.exit(1)", encoding="utf-8")

    monkeypatch.setattr(lr, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(lr, "WORKSPACES_DIR", str(tmp_path / "workspaces"))

    job = _make_job(str(dummy_h5mu), str(tmp_path / "out"))
    result = asyncio.run(lr.run_model_local(job, seed=42))
    assert result == "failed"

    conn = isolated_registry.get_db_connection()
    try:
        status, reason = conn.execute(
            "SELECT status, failure_reason FROM runs WHERE dataset_id = ?",
            (job["dataset_id"],),
        ).fetchone()
    finally:
        conn.close()
    assert status == "FAILED"
    assert reason == "LOCAL_EXIT:1"


def test_run_model_local_missing_entrypoint(tmp_path, monkeypatch, dummy_h5mu, isolated_registry):
    """run_model_local should return 'failed' and record missing entrypoint failures."""
    import multiverse.runner.local_runner as lr

    monkeypatch.setattr(lr, "MODELS_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(lr, "WORKSPACES_DIR", str(tmp_path / "workspaces"))

    job = _make_job(str(dummy_h5mu), str(tmp_path / "out"))
    result = asyncio.run(lr.run_model_local(job, seed=42))
    assert result == "failed"

    conn = isolated_registry.get_db_connection()
    try:
        status, reason = conn.execute(
            "SELECT status, failure_reason FROM runs WHERE dataset_id = ?",
            (job["dataset_id"],),
        ).fetchone()
    finally:
        conn.close()
    assert status == "FAILED"
    assert reason == "LOCAL_ENTRYPOINT_MISSING"


def test_run_jobs_locally_aggregates(tmp_path, fake_models_dir, dummy_h5mu):
    """run_jobs_locally should return a mapping of job names to statuses."""
    import multiverse.runner.local_runner as lr

    output_path = tmp_path / "output"
    job = _make_job(str(dummy_h5mu), str(output_path))
    name = f"{job['dataset_name']}_{job['model_slug']}"

    results = asyncio.run(lr.run_jobs_locally([job], seed=42))

    assert name in results
    assert results[name] == "success"


def test_mvr_worker_reads_env_vars(tmp_path, monkeypatch):
    """mvr_worker.io path constants must honour MVR_* env vars."""
    out = str(tmp_path / "custom_output")
    inp = str(tmp_path / "custom_input" / "data.h5mu")
    spec = str(tmp_path / "custom_output" / "job_spec.json")

    monkeypatch.setenv("MVR_OUTPUT_DIR", out)
    monkeypatch.setenv("MVR_INPUT_DATA_PATH", inp)
    monkeypatch.setenv("MVR_JOB_SPEC_PATH", spec)

    pytest.importorskip("anndata")

    import importlib
    import mvr_worker.io as io_mod

    importlib.reload(io_mod)

    assert io_mod.OUTPUT_DIR == out
    assert io_mod.INPUT_DATA_PATH == inp
    assert io_mod.JOB_SPEC_PATH == spec
