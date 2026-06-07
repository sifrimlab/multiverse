"""Test fixtures.

Control-plane modules (artifact, journal, simple, promotion, mvd, index,
doctor, gc, broker, registration, client) must run without ``scanpy`` /
``mudata`` / ``anndata`` installed. ML-specific fixtures live behind lazy
imports so importing this module never pulls those packages in.

A ``control_plane`` pytest marker tags fast control-plane tests so they
can be collected without the ML stack:

    pytest -m control_plane

The marker is purely declarative — collection works whether the marker
is requested or not.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "control_plane: fast tests that run without ML deps (scanpy/mudata/etc)",
    )
    config.addinivalue_line(
        "markers",
        "ml: tests that require the ml-legacy dependency group",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests that require optional external services such as Docker",
    )


# ---------------------------------------------------------------------------
# ML-stack fixtures — lazy imports, skip if absent.
# ---------------------------------------------------------------------------


def _require(module: str):
    return pytest.importorskip(
        module, reason=f"requires the ml-legacy extra ({module})"
    )


@pytest.fixture
def dummy_h5ad(tmp_path):
    sc = _require("scanpy")
    path = os.path.join(tmp_path, "test.h5ad")
    obs = pd.DataFrame(
        {"batch": ["a", "b"], "cell_type": ["c1", "c2"]},
        index=["cell1", "cell2"],
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    X = np.random.rand(2, 2)
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.write(path)
    return path


@pytest.fixture
def dummy_h5mu(tmp_path):
    sc = _require("scanpy")
    md = _require("mudata")
    path = os.path.join(tmp_path, "test.h5mu")
    obs = pd.DataFrame(
        {"batch": ["a", "b"], "cell_type": ["c1", "c2"]},
        index=["cell1", "cell2"],
    )

    rna_adata = sc.AnnData(X=np.random.rand(2, 2), obs=obs)
    atac_adata = sc.AnnData(X=np.random.rand(2, 2), obs=obs)

    mdata = md.MuData({"rna": rna_adata, "atac": atac_adata})
    mdata.obs["batch"] = ["a", "b"]
    mdata.obs["cell_type"] = ["c1", "c2"]
    mdata.write(path)
    return path


@pytest.fixture
def dummy_registry_file(tmp_path):
    path = os.path.join(tmp_path, "model_registry.json")
    data = {
        "models": [
            {"name": "pca", "docker_image": "pca:tag", "supported_omics": ["rna"]},
            {
                "name": "mofa",
                "docker_image": "mofa:tag",
                "supported_omics": ["rna", "atac"],
            },
            {
                "name": "totalvi",
                "docker_image": "totalvi:tag",
                "supported_omics": ["rna", "adt"],
            },
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with minimal anndata test data."""
    ad = _require("anndata")
    temp_dir = tempfile.mkdtemp(prefix="multiverse_test_")
    data_dir = os.path.join(temp_dir, "test_data")
    os.makedirs(data_dir, exist_ok=True)

    n_obs = 10
    n_vars = 100

    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))

    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB"], size=n_obs)
    adata.obs["batch"] = "batch1"

    adata.write_h5ad(os.path.join(data_dir, "test_rna.h5ad"))

    yield data_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def outputs_dir():
    """Setup and cleanup outputs directory."""
    output_dir = "./outputs_test/"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    yield output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
