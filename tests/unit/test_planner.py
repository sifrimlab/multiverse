
import sqlite3
import json
import os
import sys
from unittest.mock import MagicMock

# Mocking modules to avoid side effects during test
sys.modules['multiverse.logging_utils'] = MagicMock()
sys.modules['rich.live'] = MagicMock()
sys.modules['rich.table'] = MagicMock()
sys.modules['docker'] = MagicMock()

# Import the function to test
from multiverse.runner.cli import generate_execution_plan

def test_generate_execution_plan():
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

    # Scenario:
    # dataset1 (rna) is compatible with pca.
    # dataset2 (rna, atac) is compatible with pca and mofa.
    # Suppose dataset2+pca already succeeded.
    cursor.execute("INSERT INTO runs (dataset_id, model_name, status, output_path) VALUES (?, ?, ?, ?)",
                   (2, "pca", "SUCCESS", "/path/to/output"))

    # Suppose dataset1+pca failed previously.
    cursor.execute("INSERT INTO runs (dataset_id, model_name, status, output_path) VALUES (?, ?, ?, ?)",
                   (1, "pca", "FAILED", "/path/to/output2"))

    plan = generate_execution_plan(conn)

    # Expected results:
    # 1. dataset1 + pca (since it failed)
    # 2. dataset2 + mofa (since it hasn't run)

    print(f"Generated Plan: {plan}")
    assert len(plan) == 2

    model_names = [p['model_name'] for p in plan]
    dataset_names = [p['dataset_name'] for p in plan]

    assert "pca" in model_names
    assert "mofa" in model_names
    assert "dataset1" in dataset_names
    assert "dataset2" in dataset_names

    print("Test passed!")

if __name__ == "__main__":
    test_generate_execution_plan()
