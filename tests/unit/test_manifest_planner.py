import sqlite3
import json
import sys
from unittest.mock import MagicMock

# Mocking modules to avoid side effects during test
if 'multiverse.logging_utils' not in sys.modules:
    sys.modules['multiverse.logging_utils'] = MagicMock()
if 'rich.live' not in sys.modules:
    sys.modules['rich.live'] = MagicMock()
if 'rich.table' not in sys.modules:
    sys.modules['rich.table'] = MagicMock()
if 'docker' not in sys.modules:
    sys.modules['docker'] = MagicMock()

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
        "model_version TEXT, status TEXT, output_path TEXT)"
    )
    return conn


def test_generate_execution_plan_from_manifest():
    conn = _make_conn()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "dataset1", "dataset1", "/path/to/d1", json.dumps(["rna"]), "batch", "cell_type", "READY"),
    )
    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (2, "dataset2", "dataset2", "/path/to/d2", json.dumps(["rna", "atac"]), "batch", "cell_type", "READY"),
    )
    cursor.execute(
        "INSERT INTO models (slug, docker_image, supported_omics, version, status) VALUES (?, ?, ?, ?, ?)",
        ("pca", "multiverse-pca", json.dumps(["rna"]), "1.0", "ACTIVE"),
    )
    cursor.execute(
        "INSERT INTO models (slug, docker_image, supported_omics, version, status) VALUES (?, ?, ?, ?, ?)",
        ("mofa", "multiverse-mofa", json.dumps(["rna", "atac"]), "1.0", "ACTIVE"),
    )

    # dataset2+pca already succeeded
    cursor.execute(
        "INSERT INTO runs (dataset_id, model_slug, model_version, status, output_path) VALUES (?, ?, ?, ?, ?)",
        (2, "pca", "1.0", "SUCCESS", "/path/to/output"),
    )

    manifest_data = {
        "manifest_version": "1.0",
        "jobs": [
            {"dataset_id": "dataset1", "models": ["pca"]},
            {"dataset_id": "dataset2", "models": ["pca", "mofa"]},
        ],
    }

    plan = generate_execution_plan_from_manifest(conn, manifest_data)

    # Expected: dataset1+pca and dataset2+mofa; dataset2+pca is skipped
    assert len(plan) == 2

    job1 = next(j for j in plan if j["dataset_name"] == "dataset1")
    assert job1["model_name"] == "pca"

    job2 = next(j for j in plan if j["dataset_name"] == "dataset2")
    assert job2["model_name"] == "mofa"


if __name__ == "__main__":
    test_generate_execution_plan_from_manifest()
