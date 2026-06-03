"""Regression tests for issue #22 — configurable preprocessing.

Covers the manifest/spec layer (ModelManifest.preprocessing, simple-mode job
parsing) and the container-side resolver that merges per-run overrides over a
model's built-in defaults.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Model manifest defaults
# ---------------------------------------------------------------------------


def test_model_manifest_accepts_preprocessing_block():
    from multiverse.models_ingest import ModelManifest

    m = ModelManifest(
        name="demo",
        version="1.0.0",
        supported_omics=["rna"],
        runtime={"image": "multiverse-demo:1.0.0"},
        preprocessing={
            "n_top_genes": 2000,
            "log_normalization": False,
            "scale": {"rna": True},
        },
    )
    assert m.preprocessing is not None
    assert m.preprocessing.n_top_genes == 2000
    # Unset fields stay None and are dropped from the job-spec view.
    assert m.preprocessing.normalization_target_sum is None
    js = m.preprocessing.to_job_spec()
    assert js == {
        "n_top_genes": 2000,
        "log_normalization": False,
        "scale": {"rna": True},
    }


def test_model_manifest_preprocessing_optional():
    from multiverse.models_ingest import ModelManifest

    m = ModelManifest(
        name="demo",
        version="1.0.0",
        supported_omics=["rna"],
        runtime={"image": "multiverse-demo:1.0.0"},
    )
    assert m.preprocessing is None


# ---------------------------------------------------------------------------
# Simple-mode manifest parsing
# ---------------------------------------------------------------------------


def _simple_manifest_text(preprocessing_yaml: str = "") -> str:
    return (
        "schema_version: '1'\n"
        "jobs:\n"
        "  - name: demo\n"
        "    model:\n"
        "      slug: pca\n"
        "      image: multiverse-pca:1.0.0\n"
        "    dataset:\n"
        "      slug: ds\n"
        "      path: /tmp/data.h5mu\n"
        "      n_obs: 10\n"
        f"{preprocessing_yaml}"
    )


def test_simple_job_parses_preprocessing():
    from multiverse.simple.manifest import parse_simple_manifest

    text = _simple_manifest_text(
        "    preprocessing:\n"
        "      n_top_genes: 500\n"
        "      log_normalization: false\n"
    )
    manifest = parse_simple_manifest(text)
    job = manifest.jobs[0]
    assert job.preprocessing == {"n_top_genes": 500, "log_normalization": False}


def test_simple_job_preprocessing_defaults_none():
    from multiverse.simple.manifest import parse_simple_manifest

    manifest = parse_simple_manifest(_simple_manifest_text())
    assert manifest.jobs[0].preprocessing is None


def test_simple_job_preprocessing_must_be_mapping():
    from multiverse.simple.manifest import (SimpleManifestError,
                                            parse_simple_manifest)

    with pytest.raises(SimpleManifestError, match="preprocessing must be a mapping"):
        parse_simple_manifest(_simple_manifest_text("    preprocessing: 5\n"))


# ---------------------------------------------------------------------------
# Container-side resolver (needs the worker SDK's scientific deps)
# ---------------------------------------------------------------------------


def _resolver():
    pytest.importorskip("mudata")
    from mvr_worker import resolve_preprocess_params

    return resolve_preprocess_params


def test_resolver_uses_model_defaults_when_no_override():
    resolve = _resolver()
    defaults = {
        "n_top_genes": 1000,
        "scale": {"rna": False},
        "normalization_target_sum": None,
        "log_normalization": False,
    }
    out = resolve({}, ["rna"], defaults)
    assert out == defaults


def test_resolver_applies_overrides():
    resolve = _resolver()
    defaults = {
        "n_top_genes": 1000,
        "scale": {"rna": False},
        "log_normalization": False,
    }
    out = resolve(
        {"preprocessing": {"n_top_genes": 250, "log_normalization": True}},
        ["rna"],
        defaults,
    )
    assert out["n_top_genes"] == 250
    assert out["log_normalization"] is True
    # Untouched key keeps the model default.
    assert out["scale"] == {"rna": False}


def test_resolver_scale_bool_expands_to_all_modalities():
    resolve = _resolver()
    out = resolve(
        {"preprocessing": {"scale": True}},
        ["rna", "atac"],
        {"scale": {"rna": False, "atac": False}},
    )
    assert out["scale"] == {"rna": True, "atac": True}


def test_resolver_ignores_null_override_fields():
    resolve = _resolver()
    defaults = {"n_top_genes": 1000}
    out = resolve({"preprocessing": {"n_top_genes": None}}, ["rna"], defaults)
    assert out["n_top_genes"] == 1000
