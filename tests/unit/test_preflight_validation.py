"""Tests for the pre-flight validation gate in cli.py."""
import json
import os
from unittest.mock import MagicMock, patch
import pytest
import h5py
import numpy as np
import anndata as ad


def _make_job(
    dataset_id=1,
    dataset_name="ds",
    dataset_path="/data/ds.h5mu",
    model_slug="pca",
    omics_available=None,
    batch_key="batch",
    cell_type_key="cell_type",
):
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "model_name": model_slug,
        "model_slug": model_slug,
        "model_version": "1.0.0",
        "model_image": f"multiverse-{model_slug}",
        "omics_available": omics_available or ["rna"],
        "batch_key": batch_key,
        "cell_type_key": cell_type_key,
        "output_path": "/tmp/out",
        "metrics": {},
    }


def _write_h5mu(path, batch_values, cell_type_values=None):
    """Write a minimal .h5mu-like h5 file with obs columns."""
    adata = ad.AnnData(
        X=np.zeros((len(batch_values), 5)),
        obs={
            "batch": batch_values,
            **({"cell_type": cell_type_values} if cell_type_values else {}),
        },
    )
    adata.write_h5ad(path)


class TestValidatePendingJobs:
    def test_job_skipped_when_required_omics_missing(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        job = _make_job(model_slug="multivi", omics_available=["rna"])
        validated, warnings = validate_pending_jobs([job])

        skipped = [j for j in validated if j.get("_skipped")]
        runnable = [j for j in validated if not j.get("_skipped")]
        assert len(skipped) == 1
        assert len(runnable) == 0
        assert "atac" in skipped[0]["_skip_reason"]

    def test_job_passes_when_required_omics_present(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "data.h5ad")
        _write_h5mu(path, batch_values=["b1", "b2"])
        job = _make_job(model_slug="multivi", omics_available=["rna", "atac"], dataset_path=path)
        validated, warnings = validate_pending_jobs([job])

        runnable = [j for j in validated if not j.get("_skipped")]
        assert len(runnable) == 1

    def test_job_skipped_when_batch_key_missing_from_obs(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "data.h5ad")
        # Write file WITHOUT 'batch' column
        adata = ad.AnnData(X=np.zeros((5, 3)), obs={"other_col": ["x"] * 5})
        adata.write_h5ad(path)

        job = _make_job(dataset_path=path, batch_key="batch")
        validated, warnings = validate_pending_jobs([job])

        skipped = [j for j in validated if j.get("_skipped")]
        assert len(skipped) == 1
        assert "batch" in skipped[0]["_skip_reason"]

    def test_job_not_skipped_when_cell_type_missing_but_warning_issued(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "data.h5ad")
        # Write file WITH batch but WITHOUT cell_type
        adata = ad.AnnData(X=np.zeros((4, 3)), obs={"batch": ["b1", "b1", "b2", "b2"]})
        adata.write_h5ad(path)

        job = _make_job(dataset_path=path, batch_key="batch", cell_type_key="cell_type")
        validated, warnings = validate_pending_jobs([job])

        runnable = [j for j in validated if not j.get("_skipped")]
        assert len(runnable) == 1, "Job should NOT be skipped for missing cell_type"
        assert any("cell_type" in w for w in warnings), "Warning about missing cell_type expected"

    def test_single_batch_warning_issued_but_job_not_skipped(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "data.h5ad")
        # All cells in the same batch
        adata = ad.AnnData(
            X=np.zeros((4, 3)),
            obs={"batch": ["only_batch"] * 4, "cell_type": ["TypeA"] * 4},
        )
        adata.write_h5ad(path)

        job = _make_job(dataset_path=path, batch_key="batch", cell_type_key="cell_type")
        validated, warnings = validate_pending_jobs([job])

        runnable = [j for j in validated if not j.get("_skipped")]
        assert len(runnable) == 1, "Job should NOT be skipped for single batch"
        assert any("1 batch" in w for w in warnings), "Warning about single batch expected"

    def test_dataset_file_opened_once_for_multiple_jobs(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs, _read_obs_columns

        path = str(tmp_path / "data.h5ad")
        _write_h5mu(path, batch_values=["b1", "b2"])

        jobs = [
            _make_job(dataset_id=1, dataset_path=path, model_slug="pca"),
            _make_job(dataset_id=1, dataset_path=path, model_slug="mofa"),
        ]
        with patch("multiverse.runner.cli._read_obs_columns", wraps=_read_obs_columns) as mock_read:
            validate_pending_jobs(jobs)
            # Same dataset_id → file should be read only once
            assert mock_read.call_count == 1

    def test_pca_accepts_any_omics(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "data.h5ad")
        _write_h5mu(path, batch_values=["b1", "b2"])

        job = _make_job(model_slug="pca", omics_available=["rna"], dataset_path=path)
        validated, _ = validate_pending_jobs([job])
        runnable = [j for j in validated if not j.get("_skipped")]
        assert len(runnable) == 1

    def test_no_batch_key_skips_batch_check(self, tmp_path):
        """Jobs with no batch_key in registry should pass even if obs has no batch column."""
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "data.h5ad")
        adata = ad.AnnData(X=np.zeros((3, 3)), obs={"cell_type": ["A", "B", "C"]})
        adata.write_h5ad(path)

        job = _make_job(dataset_path=path, batch_key=None, cell_type_key=None)
        validated, warnings = validate_pending_jobs([job])
        runnable = [j for j in validated if not j.get("_skipped")]
        assert len(runnable) == 1


    def test_empty_anndata_is_skipped(self, tmp_path):
        from multiverse.runner.cli import validate_pending_jobs

        path = str(tmp_path / "empty.h5ad")
        adata = ad.AnnData(X=np.zeros((0, 3)), obs={"batch": [], "cell_type": []})
        adata.write_h5ad(path)

        job = _make_job(dataset_path=path, batch_key="batch", cell_type_key="cell_type")
        validated, _ = validate_pending_jobs([job])

        skipped = [j for j in validated if j.get("_skipped")]
        assert len(skipped) == 1
        assert "zero cells" in skipped[0]["_skip_reason"]
