"""Tests for multiverse.migrate_data helpers."""

from pathlib import Path

import pytest
import yaml

from multiverse.migrate_data import (
    build_dataset_yaml_dict,
    dump_dataset_yaml,
    infer_modalities_and_mapping,
    slugify_fs_safe,
    validate_dataset_yaml_content,
)


def test_slugify_fs_safe() -> None:
    assert slugify_fs_safe("PBMC 10k (Final)") == "pbmc-10k-final"
    assert slugify_fs_safe("  a  b  ") == "a-b"
    assert slugify_fs_safe("!!!") == "dataset"


def test_infer_single_h5ad() -> None:
    p = Path("/tmp/x/sample.h5ad")
    omics, raw_files, err = infer_modalities_and_mapping([p], prefer_auto=True)
    assert err is None
    assert omics == ["rna"]
    assert raw_files["rna"] == "data/sample.h5ad"


def test_infer_multi_by_filename() -> None:
    files = [
        Path("/d/rna_counts.h5ad"),
        Path("/d/atac_peaks.h5ad"),
    ]
    omics, raw_files, err = infer_modalities_and_mapping(files, prefer_auto=True)
    assert err is None
    assert set(omics) == {"atac", "rna"}
    assert raw_files["rna"] == "data/rna_counts.h5ad"
    assert raw_files["atac"] == "data/atac_peaks.h5ad"


def test_dataset_yaml_roundtrip_and_validate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "x.h5ad").write_text("stub")

    d = build_dataset_yaml_dict(
        "Test set",
        ["rna"],
        {"rna": "data/x.h5ad"},
    )
    text = dump_dataset_yaml(d)
    loaded = yaml.safe_load(text)
    errs = validate_dataset_yaml_content(loaded, tmp_path)
    assert errs == []


def test_validate_dataset_yaml_detects_missing_path(tmp_path: Path) -> None:
    d = build_dataset_yaml_dict("n", ["rna"], {"rna": "data/nope.h5ad"})
    loaded = yaml.safe_load(dump_dataset_yaml(d))
    errs = validate_dataset_yaml_content(loaded, tmp_path)
    assert any("does not exist" in e for e in errs)
