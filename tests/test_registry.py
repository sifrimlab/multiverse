import pytest
import os
import json
from multiverse.registry import load_registry, get_eligible_models, ModelEntry

@pytest.fixture
def dummy_registry_file(tmp_path):
    path = os.path.join(tmp_path, "model_registry.json")
    data = {
        "models": [
            {"name": "pca", "docker_image": "pca:tag", "supported_omics": ["rna"]},
            {"name": "mofa", "docker_image": "mofa:tag", "supported_omics": ["rna", "atac"]},
            {"name": "totalvi", "docker_image": "totalvi:tag", "supported_omics": ["rna", "adt"]}
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path

def test_load_registry(dummy_registry_file):
    registry = load_registry(dummy_registry_file)
    assert len(registry) == 3
    assert registry["pca"].docker_image == "pca:tag"
    assert "rna" in registry["pca"].supported_omics

def test_get_eligible_models(dummy_registry_file):
    registry = load_registry(dummy_registry_file)
    available_omics = ["rna", "atac"]
    user_requested = ["pca", "mofa", "totalvi"]

    eligible = get_eligible_models(user_requested, available_omics, registry)

    assert "pca" in eligible
    assert "mofa" in eligible
    assert "totalvi" not in eligible # Needs adt

def test_get_eligible_models_all(dummy_registry_file):
    registry = load_registry(dummy_registry_file)
    available_omics = ["rna", "atac", "adt"]
    user_requested = ["pca", "mofa", "totalvi"]

    eligible = get_eligible_models(user_requested, available_omics, registry)

    assert len(eligible) == 3
