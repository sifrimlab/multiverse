import asyncio
import pytest
from unittest.mock import MagicMock, patch
from multiverse.runner.docker_runner import (
    build_images_concurrently,
    run_models_concurrently,
    ResourcePool,
    InsufficientResourcesError,
    _parse_mem_gb,
    run_jobs_concurrently,
)


@pytest.mark.asyncio
async def test_build_images_concurrently():
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        # Simulate images not present locally so ensure_image_prepared falls back to pull.
        mock_client.images.get.side_effect = Exception("not found")

        image_tags = ["img1", "img2"]
        # Also patch get_db_connection so the manifest-build path is skipped.
        with patch("multiverse.runner.docker_runner.get_db_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock(cursor=MagicMock(return_value=mock_cursor)))
            mock_conn.return_value.cursor.return_value = mock_cursor
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

        def _fake_run_and_promote(client, run_kwargs, workspace_dir, final_artifact_dir, job_context=None):
            container = client.containers.run(**run_kwargs)
            wait_result = container.wait()
            exit_code = wait_result.get("StatusCode", 1)
            logs_text = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            container.remove()
            status = "SUCCESS" if exit_code == 0 else "FAILED"
            return exit_code, status, workspace_dir, logs_text

        with patch("os.path.abspath", side_effect=lambda x: x):
            with patch("os.makedirs"):
                with patch("multiverse.runner.docker_runner._write_job_spec"):
                    with patch("multiverse.runner.docker_runner.run_and_promote", side_effect=_fake_run_and_promote):
                        summary = await run_models_concurrently(
                            models_info, "data/path", 42, "output/dir"
                        )

        assert summary["model_ok"] == "success"
        assert summary["model_fail"] == "failed"
        assert mock_client.containers.run.call_count == 2
        assert mock_container1.remove.called
        assert mock_container2.remove.called


# ---------------------------------------------------------------------------
# ResourcePool unit tests
# ---------------------------------------------------------------------------

def test_parse_mem_gb():
    assert _parse_mem_gb("16g") == 16.0
    assert _parse_mem_gb("16G") == 16.0
    assert _parse_mem_gb("1024m") == pytest.approx(1.0)
    assert _parse_mem_gb(8.0) == 8.0


@pytest.mark.asyncio
async def test_resource_pool_sequential_admission():
    """With 20 GiB host and three 10 GiB jobs only two can run concurrently."""
    pool = ResourcePool(total_gb=20.0)
    admission_log: list[str] = []

    async def fake_job(name: str, gb: float, duration: float):
        await pool.acquire(gb)
        admission_log.append(f"start:{name}")
        await asyncio.sleep(duration)
        admission_log.append(f"end:{name}")
        pool.release(gb)

    # Fire all three concurrently; job-c must wait until job-a or job-b finishes.
    await asyncio.gather(
        fake_job("a", 10.0, 0.05),
        fake_job("b", 10.0, 0.05),
        fake_job("c", 10.0, 0.05),
    )

    # Both a and b are admitted before c; c can only start after one ends.
    assert admission_log.index("start:c") > min(
        admission_log.index("end:a"), admission_log.index("end:b")
    ), "job-c should not start until job-a or job-b finishes"

    # All three complete
    assert set(admission_log) == {"start:a", "end:a", "start:b", "end:b", "start:c", "end:c"}


@pytest.mark.asyncio
async def test_resource_pool_insufficient_raises():
    """A job larger than total capacity raises InsufficientResourcesError immediately."""
    pool = ResourcePool(total_gb=10.0)
    with pytest.raises(InsufficientResourcesError):
        await pool.acquire(20.0)


@pytest.mark.asyncio
async def test_run_jobs_concurrently_marks_oversized_job_failed():
    """Jobs that exceed total host RAM are marked FAILED: INSUFFICIENT_RESOURCES."""
    _jobs = [
        {
            "name": "big_job",
            "image": "img",
            "dataset_path": "/data",
            "dataset_id": 1,
            "model_name_orig": "pca",
            "model_slug": "pca",
            "model_version": "1.0.0",
            "mem_limit": "30g",
        }
    ]
    statuses: dict[str, str] = {}

    with patch("multiverse.runner.docker_runner._write_job_spec"):
        with patch("multiverse.runner.docker_runner._persist_run_status") as mock_persist:
            with patch("docker.from_env"):
                result = await run_jobs_concurrently(
                    _jobs,
                    seed=42,
                    status_callback=lambda n, s: statuses.update({n: s}),
                    host_ram_gb=20.0,
                )

    assert result["big_job"] == "failed"
    assert statuses.get("big_job") == "Failed (INSUFFICIENT_RESOURCES)"
    mock_persist.assert_called_once()
    _, kwargs = mock_persist.call_args
    # status kwarg must reflect the resource failure
    assert "INSUFFICIENT_RESOURCES" in mock_persist.call_args[1].get(
        "status", mock_persist.call_args[0][4] if mock_persist.call_args[0] else ""
    ) or any("INSUFFICIENT_RESOURCES" in str(a) for a in mock_persist.call_args[0])
