"""Tests for H4: Singularity.def scaffolds for built-in models."""
from __future__ import annotations
from pathlib import Path

import pytest
import yaml

STORE_MODELS_DIR = Path(__file__).parent.parent.parent / "store" / "models"
BUILTIN_SLUGS = ["pca", "mofa", "multivi", "mowgli", "cobolt", "totalvi"]


@pytest.mark.parametrize("slug", BUILTIN_SLUGS)
def test_singularity_def_exists(slug):
    def_path = STORE_MODELS_DIR / slug / "container" / "Singularity.def"
    assert def_path.exists(), f"Missing Singularity.def for model '{slug}'"


@pytest.mark.parametrize("slug", BUILTIN_SLUGS)
def test_singularity_def_has_required_sections(slug):
    def_path = STORE_MODELS_DIR / slug / "container" / "Singularity.def"
    content = def_path.read_text(encoding="utf-8")
    assert "Bootstrap:" in content, f"Missing Bootstrap: in {slug}/Singularity.def"
    assert "%runscript" in content, f"Missing %runscript in {slug}/Singularity.def"
    assert "%post" in content, f"Missing %post in {slug}/Singularity.def"


@pytest.mark.parametrize("slug", BUILTIN_SLUGS)
def test_model_yaml_has_apptainer_field(slug):
    yaml_path = STORE_MODELS_DIR / slug / "model.yaml"
    assert yaml_path.exists(), f"Missing model.yaml for '{slug}'"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert "apptainer" in data, f"Missing 'apptainer' key in {slug}/model.yaml"
    assert data["apptainer"].get("def_file"), f"Missing apptainer.def_file in {slug}/model.yaml"
    assert "gpu_required" in data["apptainer"], f"Missing apptainer.gpu_required in {slug}/model.yaml"


@pytest.mark.parametrize("slug", BUILTIN_SLUGS)
def test_model_yaml_validates_with_manifest(slug):
    from multiverse.models_ingest import load_model_manifest
    yaml_path = STORE_MODELS_DIR / slug / "model.yaml"
    manifest = load_model_manifest(str(yaml_path))
    assert manifest.apptainer is not None
    assert manifest.apptainer.def_file is not None
    assert isinstance(manifest.apptainer.gpu_required, bool)
