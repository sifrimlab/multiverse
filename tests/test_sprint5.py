import pytest
import os
import json
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
