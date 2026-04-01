import pytest
import scanpy as sc
import mudata as md
from multiverse.ingestion import load_dataset, validate_dataset_structure

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
