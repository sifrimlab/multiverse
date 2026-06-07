import hashlib
import json
import sqlite3
import sys
from unittest.mock import MagicMock

# Mocking modules to avoid side effects during test
if "multiverse.logging_utils" not in sys.modules:
    sys.modules["multiverse.logging_utils"] = MagicMock()
if "rich.live" not in sys.modules:
    sys.modules["rich.live"] = MagicMock()
if "rich.table" not in sys.modules:
    sys.modules["rich.table"] = MagicMock()
if "docker" not in sys.modules:
    sys.modules["docker"] = MagicMock()

from multiverse.runner.cli import generate_execution_plan_from_manifest


def _make_conn():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE datasets ("
        "id INTEGER PRIMARY KEY, name TEXT, slug TEXT, path TEXT, "
        "omics_available TEXT, batch_key TEXT, cell_type_key TEXT, status TEXT)"
    )
    cursor.execute(
        "CREATE TABLE models ("
        "slug TEXT PRIMARY KEY, docker_image TEXT, supported_omics TEXT, "
        "version TEXT, status TEXT)"
    )
    cursor.execute(
        "CREATE TABLE runs ("
        "run_id INTEGER PRIMARY KEY, dataset_id INTEGER, model_slug TEXT, "
        "model_version TEXT, status TEXT, output_path TEXT, params_hash TEXT)"
    )
    return conn


def test_generate_execution_plan_from_manifest():
    conn = _make_conn()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "dataset1",
            "dataset1",
            "/path/to/d1",
            json.dumps(["rna"]),
            "batch",
            "cell_type",
            "READY",
        ),
    )
    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            2,
            "dataset2",
            "dataset2",
            "/path/to/d2",
            json.dumps(["rna", "atac"]),
            "batch",
            "cell_type",
            "READY",
        ),
    )
    cursor.execute(
        "INSERT INTO models (slug, docker_image, supported_omics, version, status) VALUES (?, ?, ?, ?, ?)",
        ("pca", "multiverse-pca", json.dumps(["rna"]), "1.0", "ACTIVE"),
    )
    cursor.execute(
        "INSERT INTO models (slug, docker_image, supported_omics, version, status) VALUES (?, ?, ?, ?, ?)",
        ("mofa", "multiverse-mofa", json.dumps(["rna", "atac"]), "1.0", "ACTIVE"),
    )

    # dataset2+pca already succeeded with the *same* (empty) params in the
    # legacy ``runs`` table. Per STRATEGY (MVD Manifest Resume and Dedupe) the
    # planner is now a *pure* expansion: a legacy SUCCESS row must NOT suppress
    # an explicitly requested manifest job. Opt-in resume against mvd state is
    # applied later, not here.
    empty_params_hash = hashlib.sha256(
        json.dumps({}, sort_keys=True).encode()
    ).hexdigest()[:12]
    cursor.execute(
        "INSERT INTO runs (dataset_id, model_slug, model_version, status, output_path, params_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (2, "pca", "1.0", "SUCCESS", "/path/to/output", empty_params_hash),
    )

    manifest_data = {
        "manifest_version": "1.0",
        "jobs": [
            {"dataset_id": "dataset1", "models": ["pca"]},
            {"dataset_id": "dataset2", "models": ["pca", "mofa"]},
        ],
    }

    plan = generate_execution_plan_from_manifest(conn, manifest_data)

    # All three jobs are planned: the legacy SUCCESS row for dataset2+pca does
    # not drop it.
    assert len(plan) == 3
    planned = {(j["dataset_name"], j["model_name"]) for j in plan}
    assert planned == {
        ("dataset1", "pca"),
        ("dataset2", "pca"),
        ("dataset2", "mofa"),
    }

    job1 = next(j for j in plan if j["dataset_name"] == "dataset1")
    assert job1["artifact_dir_name"].startswith("manifest_dataset1_pca_")

    job_pca2 = next(
        j for j in plan if j["dataset_name"] == "dataset2" and j["model_name"] == "pca"
    )
    assert job_pca2["artifact_dir_name"].startswith("manifest_dataset2_pca_")


def _single_job_plan(job_extra: dict):
    conn = _make_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "dataset1",
            "dataset1",
            "/path/to/d1",
            json.dumps(["rna"]),
            "batch",
            "cell_type",
            "READY",
        ),
    )
    cursor.execute(
        "INSERT INTO models (slug, docker_image, supported_omics, version, status) VALUES (?, ?, ?, ?, ?)",
        ("pca", "multiverse-pca", json.dumps(["rna"]), "1.0", "ACTIVE"),
    )
    manifest_data = {
        "jobs": [{"dataset_id": "dataset1", "models": ["pca"], **job_extra}]
    }
    return generate_execution_plan_from_manifest(conn, manifest_data)


def test_plan_carries_gpu_flag_when_requested():
    # GPU opt-in (issue #30): jobs[].gpu must reach the pending job.
    plan = _single_job_plan({"gpu": True})
    assert len(plan) == 1
    assert plan[0]["gpu"] is True


def test_plan_omits_gpu_flag_by_default():
    plan = _single_job_plan({})
    assert plan[0].get("gpu") in (None, False)


def test_plan_carries_mem_limit_and_preprocessing():
    plan = _single_job_plan(
        {
            "mem_limit": "48g",
            "preprocessing": {"n_top_genes": 250, "log_normalization": True},
        }
    )
    assert plan[0]["mem_limit"] == "48g"
    assert plan[0]["preprocessing"] == {"n_top_genes": 250, "log_normalization": True}


def test_gpu_and_preprocessing_reach_executor_job_spec(tmp_path):
    # End-to-end of the option plumbing (issues #30, #22): a job dict carrying
    # gpu/preprocessing flows through build_executor_options -> _ExecutorJobSpec
    # -> the container's job_spec.json, and gpu_requested reaches the spec.
    from multiverse.mvd.docker_executor import (MvdDockerExecutor,
                                                _ExecutorJobSpec,
                                                build_executor_options)

    options = build_executor_options(
        model_slug="pca",
        model_image="multiverse-pca:1.0.0",
        dataset_slug="ds",
        dataset_path="/tmp/data.h5mu",
        dataset_n_obs=10,
        gpu_requested=True,
        preprocessing={"n_top_genes": 250, "log_normalization": True},
    )
    assert options["gpu_requested"] is True
    assert options["preprocessing"] == {"n_top_genes": 250, "log_normalization": True}

    spec = _ExecutorJobSpec.from_options(options, "attempt12345678")
    assert spec.gpu_requested is True
    assert spec.preprocessing == {"n_top_genes": 250, "log_normalization": True}

    # The job spec written into the workspace carries the preprocessing block.
    executor = MvdDockerExecutor.__new__(MvdDockerExecutor)
    executor._write_job_spec(spec, tmp_path)
    written = json.loads((tmp_path / "job_spec.json").read_text())
    assert written["preprocessing"] == {"n_top_genes": 250, "log_normalization": True}


if __name__ == "__main__":
    test_generate_execution_plan_from_manifest()
