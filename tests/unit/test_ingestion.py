import pytest
import scanpy as sc
import mudata as md
import yaml
from multiverse.ingestion import load_dataset, validate_dataset_structure, register_from_manifest
from multiverse import registry_db

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


def test_register_from_manifest_inserts_row(tmp_path, monkeypatch):
    base = tmp_path
    store = base / "store"
    datasets = store / "datasets"
    ds_dir = datasets / "pbmc-10k"
    data_dir = ds_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "RNA.h5ad").write_text("stub")

    manifest = {
        "name": "PBMC 10k",
        "omics": ["rna"],
        "raw_files": {"rna": "data/RNA.h5ad"},
        "metadata_keys": {"batch": "donor_id", "cell_type": "cell_ontology"},
    }
    manifest_path = ds_dir / "dataset.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(registry_db, "BASE_DIR", str(base))
    monkeypatch.setattr(registry_db, "DB_NAME", str(base / "mvexp_state.db"))
    monkeypatch.setattr(registry_db, "STORE_DIR", str(store))
    monkeypatch.setattr(registry_db, "DATASETS_DIR", str(datasets))
    monkeypatch.setattr(registry_db, "RAW_DATASETS_DIR", str(datasets / "raw"))
    monkeypatch.setattr(registry_db, "MODELS_DIR", str(store / "models"))
    monkeypatch.setattr(registry_db, "ARTIFACTS_DIR", str(store / "artifacts"))

    out = register_from_manifest(str(manifest_path), update=None)
    assert out["action"] in {"inserted", "updated"}

    rows = registry_db.get_all_datasets()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "PBMC 10k"
    assert row["slug"] == "pbmc-10k"
    assert row["status"] == "READY"
