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

from multiverse.runner.cli import generate_execution_plan


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


def test_generate_execution_plan():
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

    # dataset2+pca already succeeded; dataset1+pca failed
    cursor.execute(
        "INSERT INTO runs (dataset_id, model_slug, model_version, status, output_path) VALUES (?, ?, ?, ?, ?)",
        (2, "pca", "1.0", "SUCCESS", "/path/to/output"),
    )
    cursor.execute(
        "INSERT INTO runs (dataset_id, model_slug, model_version, status, output_path) VALUES (?, ?, ?, ?, ?)",
        (1, "pca", "1.0", "FAILED", "/path/to/output2"),
    )

    plan = generate_execution_plan(conn)

    # Expected: dataset1+pca (failed → retry) and dataset2+mofa (never run)
    assert len(plan) == 2

    model_names = [p["model_name"] for p in plan]
    dataset_names = [p["dataset_name"] for p in plan]

    assert "pca" in model_names
    assert "mofa" in model_names
    assert "dataset1" in dataset_names
    assert "dataset2" in dataset_names


if __name__ == "__main__":
    test_generate_execution_plan()
