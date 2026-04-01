import sqlite3
import json
import os
import sys
from unittest.mock import MagicMock
import yaml

# Mocking modules to avoid side effects during test
if 'multiverse.logging_utils' not in sys.modules:
    sys.modules['multiverse.logging_utils'] = MagicMock()
if 'rich.live' not in sys.modules:
    sys.modules['rich.live'] = MagicMock()
if 'rich.table' not in sys.modules:
    sys.modules['rich.table'] = MagicMock()
if 'docker' not in sys.modules:
    sys.modules['docker'] = MagicMock()

# Import the functions to test
from multiverse.runner.cli import generate_execution_plan_from_manifest

def test_generate_execution_plan_from_manifest():
    # Setup in-memory SQLite for testing
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create tables
    cursor.execute("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT, path TEXT, omics_available TEXT, status TEXT)")
    cursor.execute("CREATE TABLE models (name TEXT PRIMARY KEY, docker_image TEXT, supported_omics TEXT)")
    cursor.execute("CREATE TABLE runs (run_id INTEGER PRIMARY KEY, dataset_id INTEGER, model_name TEXT, status TEXT, output_path TEXT)")

    # Insert test data
    cursor.execute("INSERT INTO datasets (id, name, path, omics_available, status) VALUES (?, ?, ?, ?, ?)",
                   (1, "dataset1", "/path/to/d1", json.dumps(["rna"]), "READY"))
    cursor.execute("INSERT INTO datasets (id, name, path, omics_available, status) VALUES (?, ?, ?, ?, ?)",
                   (2, "dataset2", "/path/to/d2", json.dumps(["rna", "atac"]), "READY"))

    cursor.execute("INSERT INTO models (name, docker_image, supported_omics) VALUES (?, ?, ?)",
                   ("pca", "multiverse-pca", json.dumps(["rna"])))
    cursor.execute("INSERT INTO models (name, docker_image, supported_omics) VALUES (?, ?, ?)",
                   ("mofa", "multiverse-mofa", json.dumps(["rna", "atac"])))

    # Manifest data
    manifest_data = {
        "manifest_version": "1.0",
        "jobs": [
            {
                "dataset_id": "dataset1",
                "models": ["pca"]
            },
            {
                "dataset_id": "dataset2",
                "models": ["pca", "mofa"]
            }
        ]
    }

    # Scenario:
    # dataset2+pca already succeeded.
    cursor.execute("INSERT INTO runs (dataset_id, model_name, status, output_path) VALUES (?, ?, ?, ?)",
                   (2, "pca", "SUCCESS", "/path/to/output"))

    plan = generate_execution_plan_from_manifest(conn, manifest_data)

    # Expected results:
    # 1. dataset1 + pca
    # 2. dataset2 + mofa
    # (dataset2 + pca is skipped)

    assert len(plan) == 2

    # Check jobs
    job1 = next(j for j in plan if j["dataset_name"] == "dataset1")
    assert job1["model_name"] == "pca"

    job2 = next(j for j in plan if j["dataset_name"] == "dataset2")
    assert job2["model_name"] == "mofa"

    print("Manifest Plan Test passed!")

if __name__ == "__main__":
    test_generate_execution_plan_from_manifest()
