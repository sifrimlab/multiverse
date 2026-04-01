"""Tests for multiverse.guesser.DatasetHeuristics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from multiverse.guesser import DatasetHeuristics


class _FakeObsGroup:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def keys(self) -> list[str]:
        return list(self._keys)


class _FakeH5Root:
    """Minimal HDF5 root exposing only ``obs`` — no ``X`` or ``layers``."""

    def __init__(self, obs_keys: list[str]) -> None:
        self._obs = _FakeObsGroup(obs_keys)

    def __enter__(self) -> _FakeH5Root:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __contains__(self, name: str) -> bool:
        return name == "obs"

    def __getitem__(self, name: str) -> _FakeObsGroup:
        if name == "obs":
            return self._obs
        raise KeyError(name)


def test_shallow_peek_h5_picks_donor_and_cell_ontology() -> None:
    obs_keys = ["barcode", "donor_id", "n_genes", "cell_ontology"]
    dh = DatasetHeuristics()

    with patch(
        "multiverse.guesser.h5py.File",
        side_effect=lambda *_a, **_k: _FakeH5Root(obs_keys),
    ):
        out = dh._shallow_peek_h5(Path("/tmp/fake.h5ad"))

    assert out["batch_key"] == "donor_id"
    assert out["cell_type_key"] == "cell_ontology"
    assert "donor_id" not in (out["batch_key_alternatives"] or [])
    assert "cell_ontology" not in (out["cell_type_key_alternatives"] or [])


def test_guess_from_filenames_tags(tmp_path: Path) -> None:
    (tmp_path / "sample_rna_raw.h5ad").write_text("")
    (tmp_path / "ATAC_processed.h5ad").write_text("")
    dh = DatasetHeuristics()
    g = dh._guess_from_filenames(tmp_path)
    assert g["directory"] == str(tmp_path.resolve())
    by_name = {e["path"]: e["tags"] for e in g["files"]}
    assert "rna" in by_name["sample_rna_raw.h5ad"]
    assert "raw" in by_name["sample_rna_raw.h5ad"]
    assert "atac" in by_name["ATAC_processed.h5ad"]
    assert "processed" in by_name["ATAC_processed.h5ad"]


def test_generate_manifest_combines_lexical_and_peek(tmp_path: Path) -> None:
    (tmp_path / "RNA.h5ad").write_text("")
    obs_keys = ["barcode", "donor_id", "n_genes", "cell_ontology"]
    dh = DatasetHeuristics()

    with patch(
        "multiverse.guesser.h5py.File",
        side_effect=lambda *_a, **_k: _FakeH5Root(obs_keys),
    ):
        m = dh.generate_manifest(tmp_path)

    assert m["name"] == tmp_path.name
    assert "rna" in m["omics"]
    assert m["raw_files"]["rna"] == "data/RNA.h5ad"
    assert m["metadata_keys"]["batch"] == "donor_id"
    assert m["metadata_keys"]["cell_type"] == "cell_ontology"
