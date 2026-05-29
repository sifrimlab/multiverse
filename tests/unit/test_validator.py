import sqlite3

import anndata as ad
import numpy as np

from multiverse import registry_db
from multiverse.runner.cli import _read_obs_count, validate_pending_jobs


def _job(path):
    return {
        "dataset_id": 1,
        "dataset_name": "ds",
        "dataset_path": str(path),
        "model_name": "pca",
        "model_slug": "pca",
        "model_version": "1.0.0",
        "model_image": "img",
        "omics_available": ["rna"],
        "batch_key": "batch",
        "cell_type_key": "cell_type",
        "output_path": "/tmp/out",
    }


def _init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER, model_slug TEXT, model_version TEXT, model_name TEXT, status TEXT, output_path TEXT, failure_reason TEXT)")
    conn.commit(); conn.close()


def test_empty_anndata_records_zero_cells(tmp_path):
    data = tmp_path / "empty.h5ad"
    ad.AnnData(X=np.zeros((0, 2))).write_h5ad(data)
    assert _read_obs_count(str(data)) == 0
    validated, _ = validate_pending_jobs([_job(data)])
    assert validated[0]["_skipped"]
    assert "zero cells" in validated[0]["_skip_reason"]


def test_corrupt_h5_records_validation_error_when_enabled(tmp_path):
    db_path = str(tmp_path / "registry.db")
    _init_db(db_path)
    corrupt = tmp_path / "corrupt.h5ad"
    corrupt.write_bytes(b"not hdf5")

    old = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        validated, _ = validate_pending_jobs([_job(corrupt)], record_failures=True)
    finally:
        registry_db.DB_NAME = old

    assert validated[0]["_skipped"]
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status, failure_reason FROM runs").fetchone()
    conn.close()
    assert row == ("FAILED", "VALIDATION_ERROR:file_unreadable")


def test_missing_batch_key_records_key_mismatch(tmp_path):
    db_path = str(tmp_path / "registry.db")
    _init_db(db_path)
    data = tmp_path / "data.h5ad"
    ad.AnnData(X=np.zeros((3, 2)), obs={"other": ["a", "b", "c"]}).write_h5ad(data)

    old = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        validate_pending_jobs([_job(data)], record_failures=True)
    finally:
        registry_db.DB_NAME = old

    conn = sqlite3.connect(db_path)
    reason = conn.execute("SELECT failure_reason FROM runs").fetchone()[0]
    conn.close()
    assert reason == "VALIDATION_ERROR:missing_batch_key"
