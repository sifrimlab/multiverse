import pytest
import os
import pandas as pd
import numpy as np
import scanpy as sc
import mudata as md
import json
import tempfile
import shutil
import anndata as ad

@pytest.fixture
def dummy_h5ad(tmp_path):
    path = os.path.join(tmp_path, "test.h5ad")
    obs = pd.DataFrame({"batch": ["a", "b"], "cell_type": ["c1", "c2"]}, index=["cell1", "cell2"])
    var = pd.DataFrame(index=["gene1", "gene2"])
    X = np.random.rand(2, 2)
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.write(path)
    return path

@pytest.fixture
def dummy_h5mu(tmp_path):
    path = os.path.join(tmp_path, "test.h5mu")
    obs = pd.DataFrame({"batch": ["a", "b"], "cell_type": ["c1", "c2"]}, index=["cell1", "cell2"])

    # RNA
    rna_adata = sc.AnnData(X=np.random.rand(2, 2), obs=obs)
    # ATAC
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
            {"name": "mofa", "docker_image": "mofa:tag", "supported_omics": ["rna", "atac"]},
            {"name": "totalvi", "docker_image": "totalvi:tag", "supported_omics": ["rna", "adt"]}
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path

@pytest.fixture
def test_data_dir():
    """Create a temporary directory with minimal test data."""
    temp_dir = tempfile.mkdtemp(prefix='mvexp_test_')
    data_dir = os.path.join(temp_dir, 'test_data')
    os.makedirs(data_dir, exist_ok=True)

    n_obs = 10
    n_vars = 100

    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))

    adata = ad.AnnData(X=X)
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]

    adata.obs['cell_type'] = np.random.choice(['TypeA', 'TypeB'], size=n_obs)
    adata.obs['batch'] = 'batch1'

    adata.write_h5ad(os.path.join(data_dir, 'test_rna.h5ad'))

    yield data_dir

    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def outputs_dir():
    """Setup and cleanup outputs directory."""
    output_dir = './outputs_test/'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    yield output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
