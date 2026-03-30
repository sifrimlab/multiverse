import pytest
import os
import pandas as pd
import numpy as np
import scanpy as sc
import mudata as md
import muon as mu
from multiverse.ingestion import load_dataset, validate_dataset_structure

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

def test_load_h5ad(dummy_h5ad):
    data = load_dataset(dummy_h5ad)
    assert isinstance(data, sc.AnnData)
    assert data.n_obs == 2

def test_load_h5mu(dummy_h5mu):
    data = load_dataset(dummy_h5mu)
    assert isinstance(data, md.MuData)
    assert "rna" in data.mod
    assert "atac" in data.mod

def test_validate_structure_success(dummy_h5ad):
    data = load_dataset(dummy_h5ad)
    omics = validate_dataset_structure(data, batch_key="batch", cell_type_key="cell_type")
    assert "rna" in omics

def test_validate_structure_missing_key(dummy_h5ad):
    data = load_dataset(dummy_h5ad)
    with pytest.raises(ValueError, match="Batch key 'missing' not found"):
        validate_dataset_structure(data, batch_key="missing")

def test_validate_h5mu_omics(dummy_h5mu):
    data = load_dataset(dummy_h5mu)
    omics = validate_dataset_structure(data, batch_key="batch")
    assert set(omics) == {"rna", "atac"}
