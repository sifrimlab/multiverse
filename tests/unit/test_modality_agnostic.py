"""Regression tests for issue #25 — modality-agnostic models.

A model may declare ``supported_omics: ['any']`` to indicate it works on any
dataset regardless of modality. These tests pin both halves of the contract:

* ``ModelManifest`` accepts ``['any']`` but rejects mixing ``any`` with a
  concrete modality.
* ``generate_compatibility_matrix`` treats an ``['any']`` model as compatible
  with every dataset.
"""

from __future__ import annotations

import pytest

from multiverse.registry import generate_compatibility_matrix


def _make_manifest(omics):
    from multiverse.models_ingest import ModelManifest

    return ModelManifest(
        name="demo",
        version="1.0.0",
        supported_omics=omics,
        runtime={"image": "multiverse-demo:1.0.0"},
    )


def test_manifest_accepts_any():
    manifest = _make_manifest(["any"])
    assert manifest.supported_omics == ["any"]


def test_manifest_normalizes_case():
    manifest = _make_manifest(["ANY"])
    assert manifest.supported_omics == ["any"]


def test_manifest_rejects_any_mixed_with_modality():
    with pytest.raises(ValueError):
        _make_manifest(["any", "rna"])


def test_compatibility_matrix_any_is_compatible_with_all_datasets():
    datasets = [
        {"name": "RNA only", "omics_available": ["rna"]},
        {"name": "Multiome", "omics_available": ["rna", "atac"]},
    ]
    models = [{"name": "agnostic", "supported_omics": ["any"]}]

    matrix = generate_compatibility_matrix(datasets, models)

    assert matrix.loc["RNA only", "agnostic"] == "Compatible"
    assert matrix.loc["Multiome", "agnostic"] == "Compatible"


def test_compatibility_matrix_any_from_json_string():
    """SQLite stores supported_omics as a JSON string; ``any`` must still
    resolve to Compatible after deserialization."""
    datasets = [{"name": "RNA only", "omics_available": '["rna"]'}]
    models = [{"name": "agnostic", "supported_omics": '["any"]'}]

    matrix = generate_compatibility_matrix(datasets, models)

    assert matrix.loc["RNA only", "agnostic"] == "Compatible"
