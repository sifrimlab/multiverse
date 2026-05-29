import anndata as ad
import numpy as np

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


def test_empty_anndata_records_zero_cells(tmp_path):
    data = tmp_path / "empty.h5ad"
    ad.AnnData(X=np.zeros((0, 2))).write_h5ad(data)
    assert _read_obs_count(str(data)) == 0
    validated, _ = validate_pending_jobs([_job(data)])
    assert validated[0]["_skipped"]
    assert "zero cells" in validated[0]["_skip_reason"]


def test_corrupt_h5_skips_job(tmp_path):
    """Corrupt dataset → job is skipped with file_unreadable reason."""
    corrupt = tmp_path / "corrupt.h5ad"
    corrupt.write_bytes(b"not hdf5")
    validated, _ = validate_pending_jobs([_job(corrupt)], record_failures=True)
    assert validated[0]["_skipped"]
    assert "unreadable" in validated[0].get("_skip_reason", "").lower()


def test_missing_batch_key_skips_job(tmp_path):
    """Dataset without expected batch key → job is skipped."""
    data = tmp_path / "data.h5ad"
    ad.AnnData(X=np.zeros((3, 2)), obs={"other": ["a", "b", "c"]}).write_h5ad(data)
    validated, _ = validate_pending_jobs([_job(data)], record_failures=True)
    assert validated[0]["_skipped"]
