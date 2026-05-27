"""Optional real-Docker mvd end-to-end fault tests.

These tests exercise the full ``Kernel -> MvdDockerExecutor ->
DockerSupervisor -> RealDockerEngine -> PromotionSaga -> rebuild_index`` path.
They never pull images from the network. A developer/CI host must already have one of the candidate local base images,
or set ``MVD_REAL_DOCKER_BASE_IMAGE`` to a local shell image with ``sh`` and
``cp``. The tests never pull images implicitly.

The happy-path image copies a host-generated HDF5 fixture from the mounted
/input path into /output/embeddings.h5. That keeps the model image tiny while
still validating the real container, bind mount, promotion, manifest, and
rebuild path.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pytest

from multiverse.artifact import BootContext, read_manifest
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.docker_supervisor import DockerSupervisor, RealDockerEngine
from multiverse.index import open_index, rebuild_index
from multiverse.index.sqlite_index import INDEX_FILENAME
from multiverse.journal import JournalLayout, JournalWriter
from multiverse.mvd import Kernel, KernelConfig, MvdDockerExecutor, PrimaryState, build_executor_options
from multiverse.promotion import StoreLayout
from multiverse.promotion.saga import PromotionSaga


CANDIDATE_BASE_IMAGES = (
    "busybox:latest",
    "alpine:latest",
    "mambaorg/micromamba:2.3.0",
    "multiverse-pca:1.0.0",
)
PYTHON_BASE_CANDIDATES = (
    "multiverse-pca:1.0.0",
    "mambaorg/micromamba:2.3.0",
    "alpine:latest",
)


def _docker_client():
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")


def _docker_client_and_base_image():
    client = _docker_client()

    requested = os.environ.get("MVD_REAL_DOCKER_BASE_IMAGE")
    candidates = (requested,) if requested else CANDIDATE_BASE_IMAGES
    for image in candidates:
        if not image:
            continue
        try:
            client.images.get(image)
            return client, image
        except Exception:
            continue
    pytest.skip(
        "no local shell base image available; pre-load one of "
        f"{CANDIDATE_BASE_IMAGES!r} or set MVD_REAL_DOCKER_BASE_IMAGE"
    )


def _docker_client_and_python_base_image():
    client = _docker_client()

    requested = os.environ.get("MVD_REAL_DOCKER_OOM_BASE_IMAGE")
    candidates = (requested,) if requested else PYTHON_BASE_CANDIDATES
    for image in candidates:
        if not image:
            continue
        try:
            client.images.get(image)
            output = client.containers.run(
                image,
                ["-c", "command -v python || command -v python3"],
                entrypoint="sh",
                remove=True,
            )
            python_path = output.decode("utf-8", errors="replace").strip().splitlines()[-1]
            if python_path:
                return client, image, python_path
        except Exception:
            continue
    pytest.skip(
        "no local Python-capable base image available for OOM fixture; "
        "set MVD_REAL_DOCKER_OOM_IMAGE or MVD_REAL_DOCKER_OOM_BASE_IMAGE"
    )


def _cleanup_real_it_containers(client) -> None:
    try:
        containers = client.containers.list(
            all=True, filters={"label": "multiverse.mvd_version=0.1.0-real-it"}
        )
    except Exception:
        return
    for container in containers:
        try:
            container.remove(force=True)
        except Exception:
            pass

def _build_shell_image(client, tmp_path: Path, *, base_image: str, command: str) -> str:
    tag = f"mvd-real-path-{uuid4().hex[:12]}:latest"
    context = tmp_path / tag.replace(":", "_")
    context.mkdir()
    (context / "Dockerfile").write_text(
        "\n".join(
            [
                f"FROM {base_image}",
                "RUN mkdir -p /input /output",
                f"CMD [\"sh\", \"-c\", {json.dumps(command)}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    client.images.build(path=str(context), tag=tag, rm=True, pull=False)
    return tag


def _dataset_file(tmp_path: Path, *, n_obs: int = 4) -> Path:
    dataset = tmp_path / "fixture.h5mu"
    with h5py.File(dataset, "w") as f:
        f.create_dataset(
            "latent",
            data=np.random.default_rng(0).standard_normal((n_obs, 4)).astype(np.float32),
        )
    return dataset


def _kernel_with_real_docker(
    *,
    state_root: Path,
    store: StoreLayout,
    client,
    poll_interval_seconds: float = 0.05,
    max_poll_iterations: int = 400,
) -> Kernel:
    boot = BootContext.new(mvd_version="0.1.0-real-it")
    journal = JournalWriter(
        JournalLayout.at(state_root / "journal"),
        boot_id=boot.boot_id,
    )
    supervisor = DockerSupervisor(
        engine=RealDockerEngine(client=client),
        journal=journal,
        mvd_version="0.1.0-real-it",
    )
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=ResourceBroker(
            observer=InMemoryHostObserver(
                HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
            )
        ),
        state_root=state_root,
        poll_interval_seconds=poll_interval_seconds,
        max_poll_iterations=max_poll_iterations,
    )
    return Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-real-it"),
        executor=executor,
        journal=journal,
        boot=boot,
    )


def _opts(
    *,
    image: str,
    dataset: Path,
    n_obs: int = 4,
    mem_limit: str | None = None,
    command: list[str] | None = None,
    entrypoint: str | None = None,
) -> dict:
    return build_executor_options(
        model_slug="real-shell-model",
        model_image=image,
        image_digest=None,
        dataset_slug="demo",
        dataset_path=str(dataset),
        dataset_n_obs=n_obs,
        dataset_n_vars=8,
        params={},
        manifest_text="schema_version: '1'\n",
        artifact_dir_name=f"artifact-{uuid4().hex[:8]}",
        mem_limit=mem_limit,
        container_command=command,
        container_entrypoint=entrypoint,
    )


async def _wait_for_state(kernel: Kernel, attempt: str, state: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = await kernel.query_run(physical_attempt_id=attempt)
        if last["primary_state"] == state:
            return last
        await asyncio.sleep(0.05)
    pytest.fail(f"attempt {attempt} did not reach {state}; last={last}")


@pytest.mark.integration
def test_real_mvd_happy_path_promotes_and_rebuilds_index(tmp_path: Path) -> None:
    client, base_image = _docker_client_and_base_image()
    image = base_image
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_docker(state_root=state_root, store=store, client=client)

    async def _scenario() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=image,
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "cp /input/data.h5mu /output/embeddings.h5"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        artifact_dir = Path(snapshot["artifact_dir"])
        manifest = read_manifest(artifact_dir)
        assert manifest.physical_attempt_id == attempt
        assert {entry.name for entry in manifest.artifacts} >= {"embeddings.h5", "job_spec.json"}
        await kernel.shutdown()
        return attempt

    try:
        attempt_id = asyncio.run(_scenario())
        index_path = state_root / INDEX_FILENAME
        if index_path.exists():
            index_path.unlink()
        with open_index(index_path) as index:
            result = rebuild_index(index=index, state_root=state_root, store=store, engine=RealDockerEngine(client=client))
            row = index.get_run(attempt_id)
        assert result.artifact_success == 1
        assert row is not None
        assert row["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        assert row["artifact_dir"]
    finally:
        _cleanup_real_it_containers(client)


@pytest.mark.integration
def test_real_mvd_container_nonzero_exit_is_failed(tmp_path: Path) -> None:
    client, base_image = _docker_client_and_base_image()
    image = base_image
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_docker(state_root=state_root, store=store, client=client)

    async def _scenario() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=image,
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "exit 7"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.FAILED.value
        assert "container exited 7" in snapshot["failure_reason"]
        await kernel.shutdown()

    try:
        asyncio.run(_scenario())
    finally:
        _cleanup_real_it_containers(client)


@pytest.mark.integration
def test_real_mvd_validation_failure_quarantines_workspace(tmp_path: Path) -> None:
    client, base_image = _docker_client_and_base_image()
    image = base_image
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_docker(state_root=state_root, store=store, client=client)

    async def _scenario() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=image,
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "true"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.RECOVERY_PENDING.value
        assert snapshot["failure_reason"] == "quarantined; user adoption required"
        await kernel.shutdown()
        return attempt

    try:
        attempt_id = asyncio.run(_scenario())
        quarantines = list((store.quarantine).glob(f"*{attempt_id}*"))
        assert quarantines or any(store.quarantine.iterdir())
    finally:
        _cleanup_real_it_containers(client)


@pytest.mark.integration
def test_real_mvd_cancel_during_running_container(tmp_path: Path) -> None:
    client, base_image = _docker_client_and_base_image()
    image = base_image
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_docker(state_root=state_root, store=store, client=client)

    async def _scenario() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=image,
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "sleep 60"],
            ),
        )
        await _wait_for_state(kernel, attempt, PrimaryState.RUNNING.value)
        await kernel.cancel_run(physical_attempt_id=attempt)
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.CANCELLED.value
        await kernel.shutdown()

    try:
        asyncio.run(_scenario())
    finally:
        _cleanup_real_it_containers(client)


@pytest.mark.integration
def test_real_mvd_crash_after_promotion_stage_rebuilds_recovery_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, base_image = _docker_client_and_base_image()
    image = base_image
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_docker(state_root=state_root, store=store, client=client)

    def _crash_before_commit(self, *, staged_checksums):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated crash after promotion staging")

    monkeypatch.setattr(PromotionSaga, "_step_commit_manifest", _crash_before_commit)

    async def _scenario() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=image,
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "cp /input/data.h5mu /output/embeddings.h5"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.FAILED.value
        assert "simulated crash" in snapshot["failure_reason"]
        await kernel.shutdown()
        return attempt

    try:
        attempt_id = asyncio.run(_scenario())
        with open_index(state_root / INDEX_FILENAME) as index:
            result = rebuild_index(index=index, state_root=state_root, store=store, engine=RealDockerEngine(client=client))
            row = index.get_run(attempt_id)
        assert result.recovery_pending == 1
        assert row is not None
        assert row["primary_state"] == PrimaryState.RECOVERY_PENDING.value
        assert "promotion prepared but never committed" in row["failure_reason"]
    finally:
        _cleanup_real_it_containers(client)


@pytest.mark.integration
def test_real_mvd_oom_like_exit_classification(tmp_path: Path) -> None:
    explicit_image = os.environ.get("MVD_REAL_DOCKER_OOM_IMAGE")
    remove_image = False
    if explicit_image:
        client = _docker_client()
        try:
            client.images.get(explicit_image)
        except Exception as exc:
            pytest.skip(f"OOM image is not available locally: {exc}")
        image = explicit_image
    else:
        client, base_image, python_path = _docker_client_and_python_base_image()
        image = base_image
        remove_image = False
    oom_command = (
        f"{python_path if not explicit_image else 'python'} -c "
        + json.dumps(
            "chunks=[]\n"
            "while True:\n"
            "    chunks.append(bytearray(16 * 1024 * 1024))\n"
        )
    )

    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_docker(state_root=state_root, store=store, client=client)

    async def _scenario() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=image,
                dataset=dataset,
                n_obs=4,
                mem_limit="64m",
                entrypoint="sh",
                command=["-c", oom_command],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.FAILED.value
        assert "OOM" in snapshot["failure_reason"]
        await kernel.shutdown()

    try:
        asyncio.run(_scenario())
    finally:
        _cleanup_real_it_containers(client)
        if remove_image:
            try:
                client.images.remove(image=image, force=True)
            except Exception:
                pass
