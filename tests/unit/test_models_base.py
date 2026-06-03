import os

import anndata as ad
import h5py
import numpy as np
import pytest

from multiverse.models.base import ModelFactory


def test_save_latent_atomic(tmp_path):
    # Setup dummy dataset and model factory
    dataset = ad.AnnData(X=np.zeros((10, 10)))
    dataset.obsm["X_test_model"] = np.random.rand(10, 2)

    config = {"output_dir": str(tmp_path), "model": {"test_model": {}}}

    model = ModelFactory(
        dataset=dataset,
        dataset_name="test_dataset",
        model_name="test_model",
        config_path=config,
    )

    # Path where latent should be saved
    expected_path = os.path.join(
        tmp_path, "test_dataset", "test_model", "embeddings.h5"
    )

    # Call save_latent
    model.save_latent()

    # Verify file exists and has correct content
    assert os.path.exists(expected_path)
    with h5py.File(expected_path, "r") as f:
        assert "latent" in f
        saved_latent = f["latent"][:]
        np.testing.assert_array_equal(saved_latent, dataset.obsm["X_test_model"])

    # Verify tmp file is gone
    assert not os.path.exists(f"{expected_path}.tmp")


def test_save_latent_cleanup_on_failure(tmp_path, monkeypatch):
    # Setup dummy dataset and model factory
    dataset = ad.AnnData(X=np.zeros((10, 10)))
    dataset.obsm["X_test_model"] = np.random.rand(10, 2)

    config = {"output_dir": str(tmp_path), "model": {"test_model": {}}}

    model = ModelFactory(
        dataset=dataset,
        dataset_name="test_dataset",
        model_name="test_model",
        config_path=config,
    )

    # Mock os.rename to raise an exception to simulate failure after writing tmp file
    def mock_rename(src, dst):
        raise OSError("Simulated rename failure")

    monkeypatch.setattr(os, "rename", mock_rename)

    expected_path = os.path.join(
        tmp_path, "test_dataset", "test_model", "embeddings.h5"
    )
    tmp_path_file = f"{expected_path}.tmp"

    with pytest.raises(OSError, match="Simulated rename failure"):
        model.save_latent()

    # Verify final file does NOT exist
    assert not os.path.exists(expected_path)
    # Verify tmp file was cleaned up
    assert not os.path.exists(tmp_path_file)
