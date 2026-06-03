"""Tests for multiverse.migrate_data orchestration helpers."""

from pathlib import Path

from multiverse.migrate_data import (_render_yaml_with_notes, _slugify_relpath,
                                     safe_copy_file, slugify_fs_safe)


def test_slugify_fs_safe() -> None:
    assert slugify_fs_safe("PBMC 10k (Final)") == "pbmc-10k-final"
    assert slugify_fs_safe("  a  b  ") == "a-b"
    assert slugify_fs_safe("!!!") == "dataset"


def test_slugify_relpath_preserves_structure() -> None:
    rel = Path("Donor A/Run 01")
    assert _slugify_relpath(rel, fallback="x") == "donor-a-run-01"


def test_render_yaml_with_alternative_comments() -> None:
    manifest = {
        "name": "x",
        "omics": ["rna"],
        "raw_files": {"rna": "data/RNA.h5ad"},
        "metadata_keys": {"batch": "batch", "cell_type": "cell_type"},
        "guesser_notes": {
            "peek": {
                "batch_key_alternatives": ["donor_id"],
                "cell_type_key_alternatives": ["cell_ontology"],
            }
        },
    }
    text = _render_yaml_with_notes(manifest)
    assert text.startswith("# Heuristic alternatives found")
    assert "batch alternatives: donor_id" in text
    assert "cell_type alternatives: cell_ontology" in text


def test_safe_copy_file_fallback_copy(tmp_path: Path) -> None:
    src = tmp_path / "a.h5ad"
    dst = tmp_path / "out" / "a.h5ad"
    src.write_text("hello")
    method = safe_copy_file(src, dst)
    assert dst.exists()
    assert method in {"hardlink", "copy2"}
