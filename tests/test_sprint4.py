import pytest
import asyncio
from unittest.mock import MagicMock, patch
from multiverse.runner.docker_runner import build_images_concurrently, run_models_concurrently


@pytest.mark.asyncio
async def test_build_images_concurrently():
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client

        image_tags = ["img1", "img2"]
        await build_images_concurrently(image_tags)

        assert mock_client.images.pull.call_count == 2
        mock_client.images.pull.assert_any_call("img1")
        mock_client.images.pull.assert_any_call("img2")


@pytest.mark.asyncio
async def test_run_models_concurrently_success_and_failure():
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client

        # Mocking container 1 (success)
        mock_container1 = MagicMock()
        mock_container1.wait.return_value = {"StatusCode": 0}
        mock_container1.logs.return_value = b"All good"

        # Mocking container 2 (failure)
        mock_container2 = MagicMock()
        mock_container2.wait.return_value = {"StatusCode": 1}
        mock_container2.logs.return_value = b"Out of memory"

        # side_effect for containers.run
        mock_client.containers.run.side_effect = [mock_container1, mock_container2]

        models_info = [
            {"name": "model_ok", "image": "img_ok"},
            {"name": "model_fail", "image": "img_fail"}
        ]

        # We need a real path for os.path.abspath if not careful, but let's just mock it
        with patch("os.path.abspath", side_effect=lambda x: x):
            with patch("os.makedirs"):
                summary = await run_models_concurrently(
                    models_info, "data/path", 42, "output/dir"
                )

        assert summary["model_ok"] == "success"
        assert summary["model_fail"] == "failed"
        assert mock_client.containers.run.call_count == 2
        assert mock_container1.remove.called
        assert mock_container2.remove.called
