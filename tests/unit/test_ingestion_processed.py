"""Regression tests for issue #23 — processed-dataset registration.

A dataset manifest may now register an already-processed ``.h5mu``/``.h5ad``
via ``processed_path`` instead of requiring ``raw_files``. These tests avoid
importing scanpy/mudata: ``register_from_manifest`` is metadata-only and loads
no scientific data.
"""

from __future__ import annotations

import pytest
import yaml

from multiverse import registry_db
from multiverse.ingestion import DatasetManifest, register_from_manifest


def _isolate(monkeypatch, base):
    store = base / "store"
    datasets = store / "datasets"
    for attr, path in {
        "BASE_DIR": base,
        "DB_NAME": base / "mvexp_state.db",
        "STORE_DIR": store,
        "DATASETS_DIR": datasets,
        "RAW_DATASETS_DIR": datasets / "raw",
        "MODELS_DIR": store / "models",
        "ARTIFACTS_DIR": store / "artifacts",
    }.items():
        monkeypatch.setattr(registry_db, attr, str(path))
    return datasets


def test_manifest_requires_raw_or_processed():
    with pytest.raises(ValueError, match="raw_files.*processed_path"):
        DatasetManifest(name="d", omics=["rna"])


def test_manifest_accepts_processed_path_only():
    m = DatasetManifest(name="d", omics=["rna"], processed_path="data/processed.h5mu")
    assert m.processed_path == "data/processed.h5mu"
    assert m.raw_files == {}


def test_manifest_rejects_both_raw_and_processed():
    # The two registration modes are mutually exclusive (issue #23): a manifest
    # carrying both raw_files and processed_path is ambiguous and rejected.
    with pytest.raises(ValueError, match="exactly one"):
        DatasetManifest(
            name="d",
            omics=["rna"],
            raw_files={"rna": "data/rna.h5ad"},
            processed_path="data/processed.h5mu",
        )


def test_register_processed_dataset(tmp_path, monkeypatch):
    datasets = _isolate(monkeypatch, tmp_path)
    ds_dir = datasets / "proc-ds"
    data_dir = ds_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "processed.h5mu").write_text("stub")

    manifest = {
        "name": "Processed DS",
        "omics": ["rna"],
        "processed_path": "data/processed.h5mu",
    }
    manifest_path = ds_dir / "dataset.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    out = register_from_manifest(str(manifest_path), update=None)
    assert out["action"] in {"inserted", "updated"}

    rows = registry_db.get_all_datasets()
    assert len(rows) == 1
    assert rows[0]["status"] == "READY"
    # The registered path points at the processed file the user supplied.
    assert rows[0]["path"].endswith("data/processed.h5mu")


def test_register_processed_dataset_missing_file_errors(tmp_path, monkeypatch):
    datasets = _isolate(monkeypatch, tmp_path)
    ds_dir = datasets / "missing-proc"
    ds_dir.mkdir(parents=True)
    manifest = {
        "name": "Missing",
        "omics": ["rna"],
        "processed_path": "data/nope.h5mu",
    }
    manifest_path = ds_dir / "dataset.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="processed_path"):
        register_from_manifest(str(manifest_path), update=None)
