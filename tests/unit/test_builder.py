from unittest.mock import MagicMock, patch

from multiverse.builder import build_local_model
from multiverse.models_ingest import BuildSpec, ModelManifest, RuntimeSpec


def test_build_local_model_calls_docker_with_expected_path_and_tag(tmp_path):
    model_dir = tmp_path / "store" / "models" / "pca"
    context_dir = model_dir / "container"
    context_dir.mkdir(parents=True)
    dockerfile = context_dir / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    manifest = ModelManifest(
        name="pca",
        version="1.0.0",
        supported_omics=["rna"],
        runtime=RuntimeSpec(image="local/pca:dev"),
        build=BuildSpec(context="container", dockerfile="Dockerfile"),
        manifest_path=str(model_dir / "model.yaml"),
    )

    fake_client = MagicMock()
    fake_image = MagicMock()
    fake_image.short_id = "sha256:abc123"
    fake_logs = [{"stream": "Step 1/1\n"}]
    fake_client.images.build.return_value = (fake_image, fake_logs)

    with patch("docker.from_env", return_value=fake_client):
        build_local_model(manifest)

    call_kwargs = fake_client.images.build.call_args
    assert call_kwargs is not None, "images.build was never called"
    kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert kwargs.get("tag") == "local/pca:dev"
    assert kwargs.get("dockerfile") == "Dockerfile"
    assert kwargs.get("rm") is True
    assert kwargs.get("pull") is False
