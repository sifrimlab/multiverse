"""Tests for metrics config propagation through job_spec and per-model enforcement."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_build_model_config_returns_metrics():
    from multiverse.models.runtime_io import build_model_config

    spec = {
        "seed": 7,
        "hyperparameters": {"pca": {"n_components": 10}},
        "metrics": {"model_metrics": ["total_variance"]},
    }
    cfg = build_model_config("pca", spec, output_dir="/output")
    assert cfg["metrics"] == {"model_metrics": ["total_variance"]}


def test_build_model_config_metrics_defaults_to_empty_dict():
    from multiverse.models.runtime_io import build_model_config

    spec = {"seed": 7, "hyperparameters": {}}
    cfg = build_model_config("pca", spec, output_dir="/output")
    assert cfg["metrics"] == {}


# ── manifest metrics propagation ─────────────────────────────────────────────


def test_manifest_global_metrics_propagated_to_jobs(tmp_path):
    from unittest.mock import MagicMock

    from multiverse.runner.cli import generate_execution_plan_from_manifest

    manifest = {
        "globals": {
            "metrics": {
                "bio_conservation": ["silhouette_label"],
                "batch_correction": [],
            }
        },
        "jobs": [{"dataset_slug": "ds1", "model_name": "pca"}],
    }

    # Minimal DB mock
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        (1, "ds1", "/data/ds1.h5mu", '["rna"]', "batch", "cell_type"),  # dataset
        ("multiverse-pca", "pca", "1.0.0"),  # model
        None,  # no existing run
    ]
    conn = MagicMock()
    conn.cursor.return_value = cursor

    jobs = generate_execution_plan_from_manifest(conn, manifest)
    assert len(jobs) == 1
    assert jobs[0]["metrics"]["bio_conservation"] == ["silhouette_label"]


def test_per_job_metrics_override_globals(tmp_path):
    from unittest.mock import MagicMock

    from multiverse.runner.cli import generate_execution_plan_from_manifest

    manifest = {
        "globals": {
            "metrics": {"model_metrics": ["total_variance", "silhouette_score"]}
        },
        "jobs": [
            {
                "dataset_slug": "ds1",
                "model_name": "pca",
                "metrics": {"model_metrics": ["total_variance"]},  # override
            }
        ],
    }

    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        (1, "ds1", "/data/ds1.h5mu", '["rna"]', "batch", "cell_type"),
        ("multiverse-pca", "pca", "1.0.0"),
        None,
    ]
    conn = MagicMock()
    conn.cursor.return_value = cursor

    jobs = generate_execution_plan_from_manifest(conn, manifest)
    assert jobs[0]["metrics"]["model_metrics"] == ["total_variance"]


# ── per-model metric enforcement ─────────────────────────────────────────────


def test_pca_skips_metric_when_not_in_requested(tmp_path):
    import anndata as ad

    from multiverse.models.pca import PCAModel

    adata = ad.AnnData(X=np.eye(5))
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs_names = [f"c{i}" for i in range(5)]
    adata.obsm["X_pca"] = np.random.rand(5, 2)
    adata.uns["pca"] = {"variance_ratio": np.array([0.4, 0.2])}

    config = {
        "output_dir": str(tmp_path),
        "seed": 42,
        "model": {
            "pca": {
                "n_components": 2,
                "device": "cpu",
                "umap_random_state": 42,
                "umap_color_type": "cell_type",
            }
        },
        "metrics": {"model_metrics": []},  # empty → no metrics
    }
    model = PCAModel(dataset=adata, dataset_name="test", config_path=config)
    model.variance_ratio = [0.4, 0.2]
    model.evaluate_model()

    with open(model.metrics_filepath) as f:
        saved = json.load(f)
    assert "total_variance" not in saved


def test_pca_computes_metric_when_requested(tmp_path):
    import anndata as ad

    from multiverse.models.pca import PCAModel

    adata = ad.AnnData(X=np.eye(5))
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs_names = [f"c{i}" for i in range(5)]
    adata.obsm["X_pca"] = np.random.rand(5, 2)
    adata.uns["pca"] = {"variance_ratio": np.array([0.4, 0.2])}

    config = {
        "output_dir": str(tmp_path),
        "seed": 42,
        "model": {
            "pca": {
                "n_components": 2,
                "device": "cpu",
                "umap_random_state": 42,
                "umap_color_type": "cell_type",
            }
        },
        "metrics": {"model_metrics": ["total_variance"]},
    }
    model = PCAModel(dataset=adata, dataset_name="test", config_path=config)
    model.variance_ratio = [0.4, 0.2]
    model.evaluate_model()

    with open(model.metrics_filepath) as f:
        saved = json.load(f)
    assert "total_variance" in saved
    assert abs(saved["total_variance"] - 0.6) < 1e-6


def test_pca_computes_all_defaults_when_no_metrics_config(tmp_path):
    import anndata as ad

    from multiverse.models.pca import PCAModel

    adata = ad.AnnData(X=np.eye(5))
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs_names = [f"c{i}" for i in range(5)]
    adata.obsm["X_pca"] = np.random.rand(5, 2)
    adata.uns["pca"] = {"variance_ratio": np.array([0.4, 0.2])}

    config = {
        "output_dir": str(tmp_path),
        "seed": 42,
        "model": {
            "pca": {
                "n_components": 2,
                "device": "cpu",
                "umap_random_state": 42,
                "umap_color_type": "cell_type",
            }
        },
        # no "metrics" key → all defaults computed
    }
    model = PCAModel(dataset=adata, dataset_name="test", config_path=config)
    model.variance_ratio = [0.4, 0.2]
    model.evaluate_model()

    with open(model.metrics_filepath) as f:
        saved = json.load(f)
    assert "total_variance" in saved
