import pytest
import os
import json
import numpy as np
import h5py
import pandas as pd
import anndata as ad
from multiverse.evaluate import determine_valid_metrics, aggregate_results

def test_determine_valid_metrics_no_label():
    config = {"batch_key": "batch", "cell_type_key": None}
    dataset = ad.AnnData(obs=pd.DataFrame({"batch": ["b1", "b2", "b1", "b2"]}))

    requested = {
        "bio_conservation": ["nmi", "ari", "silhouette"],
        "batch_correction": ["graph_connectivity"]
    }

    valid = determine_valid_metrics(config, dataset, requested)

    # ARI and NMI should be removed
    assert "ari" not in valid["bio_conservation"]
    assert "nmi" not in valid["bio_conservation"]
    assert "silhouette" in valid["bio_conservation"]
    assert "graph_connectivity" in valid["batch_correction"]

def test_determine_valid_metrics_one_batch():
    config = {"batch_key": "batch", "cell_type_key": "cell_type"}
    dataset = ad.AnnData(obs=pd.DataFrame({
        "batch": ["b1", "b1", "b1", "b1"],
        "cell_type": ["c1", "c2", "c1", "c2"]
    }))

    requested = {
        "bio_conservation": ["silhouette"],
        "batch_correction": ["graph_connectivity"]
    }

    valid = determine_valid_metrics(config, dataset, requested)

    assert "silhouette" in valid["bio_conservation"]
    # graph_connectivity should be removed because num_batches == 1
    assert "graph_connectivity" not in valid["batch_correction"]

def test_aggregate_results(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    model_ok_dir = output_dir / "model_ok"
    model_ok_dir.mkdir()
    with open(model_ok_dir / "metrics.json", "w") as f:
        json.dump({"score": 0.9}, f)

    model_fail_dir = output_dir / "model_fail"
    model_fail_dir.mkdir() # shouldn't be read anyway

    model_status = {
        "model_ok": "success",
        "model_fail": "failed"
    }

    results = aggregate_results(model_status, str(output_dir))

    assert "model_ok" in results
    assert "model_fail" not in results
    assert results["model_ok"]["score"] == 0.9

    assert os.path.exists(output_dir / "results.json")


def test_evaluate_single_run(tmp_path):
    # Setup dummy dataset
    dataset_path = tmp_path / "test_dataset.h5ad"
    obs = pd.DataFrame({
        "batch": ["b1", "b2", "b1", "b2"] * 10,
        "cell_type": ["c1", "c2", "c1", "c2"] * 10
    })
    dataset = ad.AnnData(X=np.random.rand(40, 10), obs=obs)
    dataset.write_h5ad(dataset_path)

    # Setup dummy output dir with embeddings.h5
    output_dir = tmp_path / "model_output"
    output_dir.mkdir()
    embeddings_path = output_dir / "embeddings.h5"
    with h5py.File(embeddings_path, "w") as f:
        f.create_dataset("latent", data=np.random.rand(40, 2))

    from multiverse.evaluate import evaluate_single_run

    # Mock Benchmarker to avoid real heavy computation and failures
    from unittest.mock import patch, MagicMock

    with patch("multiverse.evaluate.Benchmarker") as mock_benchmarker:
        mock_instance = MagicMock()
        mock_benchmarker.return_value = mock_instance

        # Simulate benchmark results
        mock_instance.get_results.return_value = pd.DataFrame({
            "Metric": ["ARI", "NMI"],
            "X_latent": [0.8, 0.9]
        }).set_index("Metric")

        # Run evaluation
        metrics = evaluate_single_run(
            output_dir=str(output_dir),
            dataset_path=str(dataset_path),
            batch_key="batch",
            label_key="cell_type"
        )

    # Check that metrics were calculated and saved
    assert metrics
    assert "ARI" in metrics["X_latent"]
    assert os.path.exists(output_dir / "metrics.json")
    with open(output_dir / "metrics.json", "r") as f:
        saved_metrics = json.load(f)
    assert saved_metrics == metrics
