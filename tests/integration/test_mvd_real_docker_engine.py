"""Real Docker smoke tests for the mvd container adapter.

These tests are skipped unless a local Docker daemon is reachable and a tiny
shell image is already available. They intentionally do not pull from the
network; CI or a developer can pre-load one of the candidate local images to
exercise the real adapter.
"""

from __future__ import annotations

import importlib
import time
from uuid import uuid4

import pytest

from multiverse.docker_supervisor import ContainerState, RealDockerEngine

CANDIDATE_IMAGES = (
    "busybox:latest",
    "alpine:latest",
    "mambaorg/micromamba:2.3.0",
    "multiverse-pca:1.0.0",
)


def _docker_client_and_image():
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")
    for image in CANDIDATE_IMAGES:
        try:
            client.images.get(image)
            return client, image
        except Exception:
            continue
    pytest.skip(
        f"no local candidate image available from {CANDIDATE_IMAGES!r}; not pulling in tests"
    )


def _wait_for_exit(engine: RealDockerEngine, container_id: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = engine.inspect(container_id)
        if last.state is ContainerState.EXITED:
            return last
        time.sleep(0.1)
    pytest.fail(f"container did not exit before timeout; last={last}")


@pytest.mark.integration
def test_real_docker_engine_observes_nonzero_exit() -> None:
    client, image = _docker_client_and_image()
    engine = RealDockerEngine(client=client)
    name = f"mvd-it-{uuid4().hex[:10]}"
    info = engine.launch(
        image=image,
        command=["-c", "exit 7"],
        labels={"multiverse.test": name},
        env={},
        volumes={},
        mem_limit=None,
        name=name,
        entrypoint="sh",
    )
    try:
        exited = _wait_for_exit(engine, info.container_id)
        assert exited.state is ContainerState.EXITED
        assert exited.exit_code == 7
    finally:
        engine.remove(info.container_id, force=True)


@pytest.mark.integration
def test_real_docker_engine_stop_transitions_container_to_exited() -> None:
    client, image = _docker_client_and_image()
    engine = RealDockerEngine(client=client)
    name = f"mvd-it-{uuid4().hex[:10]}"
    info = engine.launch(
        image=image,
        command=["-c", "sleep 30"],
        labels={"multiverse.test": name},
        env={},
        volumes={},
        mem_limit=None,
        name=name,
        entrypoint="sh",
    )
    try:
        assert engine.inspect(info.container_id).state in {
            ContainerState.PENDING,
            ContainerState.RUNNING,
        }
        engine.stop(info.container_id, timeout=1)
        exited = _wait_for_exit(engine, info.container_id)
        assert exited.state is ContainerState.EXITED
    finally:
        engine.remove(info.container_id, force=True)
