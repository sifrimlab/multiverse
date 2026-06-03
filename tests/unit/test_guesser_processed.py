"""Tests for processed-dataset detection in the guesser / migrate tool (#23).

When a source directory already contains a processed ``.h5mu``/``.h5ad``, the
heuristics should emit a ``processed_path`` manifest (mutually exclusive with
``raw_files``) so registration skips the raw-ingestion / preprocessing step.
"""

from __future__ import annotations

from pathlib import Path

import h5py

from multiverse.guesser import DatasetHeuristics


def _write_processed_h5mu(path: Path, modalities=("rna", "atac")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.create_group("batch")
        obs.create_group("cell_type")
        mod = f.create_group("mod")
        for m in modalities:
            mod.create_group(m)


def test_guesser_emits_processed_path_for_processed_h5mu(tmp_path: Path):
    ds = tmp_path / "mydataset"
    _write_processed_h5mu(ds / "data" / "processed.h5mu", modalities=("rna", "atac"))

    manifest = DatasetHeuristics().generate_manifest(ds)

    assert manifest.get("processed_path") == "data/processed.h5mu"
    assert "raw_files" not in manifest
    assert manifest["omics"] == ["atac", "rna"]
    assert manifest["guesser_notes"]["mode"] == "processed"


def test_guesser_processed_h5ad_is_rna(tmp_path: Path):
    ds = tmp_path / "rnaonly"
    (ds).mkdir(parents=True)
    with h5py.File(ds / "processed.h5ad", "w") as f:
        f.create_group("obs")

    manifest = DatasetHeuristics().generate_manifest(ds)
    assert manifest["processed_path"] == "processed.h5ad"
    assert manifest["omics"] == ["rna"]


def test_guesser_still_emits_raw_files_without_processed(tmp_path: Path):
    ds = tmp_path / "rawds"
    ds.mkdir(parents=True)
    # A non-"processed" h5ad at the dataset root is treated as a raw modality file.
    with h5py.File(ds / "rna.h5ad", "w") as f:
        f.create_group("obs")

    manifest = DatasetHeuristics().generate_manifest(ds)
    assert "processed_path" not in manifest
    assert manifest.get("raw_files")
    assert manifest["guesser_notes"]["mode"] == "raw"


def test_migrate_rewrites_processed_path_to_data_subdir(tmp_path: Path):
    from multiverse import migrate_data

    src_root = tmp_path / "src"
    src_dir = src_root / "ds1"
    # Processed file sits at the dataset root in the source layout.
    _write_processed_h5mu(src_dir / "processed.h5mu")
    dest = tmp_path / "store" / "datasets"

    result = migrate_data.migrate_one(
        src_root, src_dir, dest, migrate_data.DatasetHeuristics(), dry_run=False
    )

    assert result.status in {"migrated", "verify"}, result.message
    written = (result.dest_dataset_dir / "dataset.yaml").read_text()
    # Migration flattens files under data/, so processed_path must follow.
    assert "processed_path: data/processed.h5mu" in written
    assert (result.dest_dataset_dir / "data" / "processed.h5mu").is_file()
