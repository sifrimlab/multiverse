from multiverse.registry import load_registry, get_eligible_models

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
