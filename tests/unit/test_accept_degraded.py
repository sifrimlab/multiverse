"""G1 acceptance tests — enforce ``accept_degraded=False`` (STRATEGY G1).

Default-config runs with a non-strict-acceptable image identity must
transition to FAILED *before* the engine receives the job. A
strict-acceptable identity (registry_digest) is always admitted.
``accept_degraded=True`` opts in to the old permissive behaviour.

Tests are parametrized over Docker and Slurm executors to satisfy the
acceptance criterion: "parametrize across Docker + Slurm executors."
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pytest

from multiverse.artifact import BootContext
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.docker_supervisor import (DockerSupervisor,
                                          InMemoryContainerEngine)
from multiverse.journal import JournalLayout, JournalWriter
from multiverse.mvd import (Kernel, KernelConfig, MvdDockerExecutor,
                            MvdSlurmExecutor, PrimaryState,
                            build_executor_options,
                            build_slurm_executor_options)
from multiverse.promotion import StoreLayout
from multiverse.slurm import InMemorySlurmEngine

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _journal(state_root: Path, boot_id: str) -> JournalWriter:
    return JournalWriter(
        JournalLayout.at(state_root / "journal").ensure(), boot_id=boot_id
    )


def _good_producer(workspace: Path, _params: Mapping[str, Any]) -> None:
    with h5py.File(workspace / "embeddings.h5", "w") as f:
        f.create_dataset("latent", data=np.zeros((4, 4), dtype=np.float32))


def _make_docker_producer(engine: InMemoryContainerEngine):
    """Return a producer that writes outputs AND exits the in-memory container.

    The InMemoryContainerEngine starts containers in RUNNING state; the
    executor's poll loop will spin forever unless the producer also flips
    the state to EXITED (same pattern used by test_mvd_docker_executor.py).
    """

    def _producer(workspace: Path, params: Mapping[str, Any]) -> None:
        _good_producer(workspace, params)
        for c in reversed(list(engine.containers.values())):
            if not c.removed and c.state.value == "running":
                engine.simulate_natural_exit(c.container_id, exit_code=0)
                return

    return _producer


# ---------------------------------------------------------------------------
# Docker executor factory
# ---------------------------------------------------------------------------


def _docker_executor(
    state_root: Path,
    store: StoreLayout,
    *,
    accept_degraded: bool,
) -> tuple[MvdDockerExecutor, JournalWriter]:
    boot = BootContext.new(mvd_version="0.1.0-test")
    writer = _journal(state_root, boot.boot_id)
    engine = InMemoryContainerEngine()
    supervisor = DockerSupervisor(
        engine=engine, journal=writer, mvd_version="0.1.0-test"
    )
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        )
    )
    executor = MvdDockerExecutor(
        journal=writer,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=broker,
        state_root=state_root,
        producer_hook=_make_docker_producer(engine),
        poll_interval_seconds=0.0,
        max_poll_iterations=200,
        accept_degraded=accept_degraded,
    )
    return executor, writer


def _docker_kernel(
    state_root: Path, store: StoreLayout, *, accept_degraded: bool
) -> tuple[Kernel, MvdDockerExecutor]:
    executor, writer = _docker_executor(
        state_root, store, accept_degraded=accept_degraded
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=writer,
        boot=executor.boot,
        broker=executor.broker,
    )
    return kernel, executor


def _docker_opts(*, dataset: Path, image_digest: str | None) -> dict:
    return build_executor_options(
        model_slug="pca",
        model_image="pca:local",
        image_digest=image_digest,
        dataset_slug="demo",
        dataset_path=str(dataset),
        dataset_n_obs=4,
        params={"n_components": 4},
    )


# ---------------------------------------------------------------------------
# Slurm executor factory
# ---------------------------------------------------------------------------


class _DrivingSlurmExecutor(MvdSlurmExecutor):
    """Self-driving Slurm executor — completes the job right after submit."""

    def __init__(self, *, n_obs: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_obs = n_obs

    def _record_dispatch(self, attempt_id: str, job_id: str) -> None:
        super()._record_dispatch(attempt_id, job_id)
        workspace_dir = self.store.workspaces / attempt_id

        def _drive() -> None:
            import time as _time

            _time.sleep(0.01)
            assert isinstance(self.engine, InMemorySlurmEngine)
            self.engine.simulate_running(job_id)
            _good_producer(workspace_dir, {})
            self.engine.simulate_completed(job_id, exit_code=0)

        threading.Thread(target=_drive, daemon=True).start()


def _slurm_executor(
    state_root: Path,
    store: StoreLayout,
    *,
    accept_degraded: bool,
) -> tuple[_DrivingSlurmExecutor, JournalWriter]:
    boot = BootContext.new(mvd_version="0.1.0-test")
    writer = _journal(state_root, boot.boot_id)
    engine = InMemorySlurmEngine()
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=1, ram_total_bytes=1024)
        ),
        max_inflight_dispatches=8,
        journal=writer,
    )
    executor = _DrivingSlurmExecutor(
        journal=writer,
        boot=boot,
        store=store,
        engine=engine,
        broker=broker,
        state_root=state_root,
        poll_interval_seconds=0.01,
        max_poll_iterations=200,
        accept_degraded=accept_degraded,
    )
    return executor, writer


def _slurm_kernel(
    state_root: Path, store: StoreLayout, *, accept_degraded: bool
) -> tuple[Kernel, _DrivingSlurmExecutor]:
    executor, writer = _slurm_executor(
        state_root, store, accept_degraded=accept_degraded
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=writer,
        boot=executor.boot,
        broker=executor.broker,
    )
    return kernel, executor


def _slurm_opts(*, dataset: Path, sif: Path, image_digest: str | None) -> dict:
    return build_slurm_executor_options(
        model_slug="pca",
        image_sif=str(sif),
        image_digest=image_digest,
        dataset_slug="demo",
        dataset_path=str(dataset),
        dataset_n_obs=4,
        params={"n_components": 4},
        partition="cpu",
        time_minutes=5,
    )


# ---------------------------------------------------------------------------
# Parametrize: ("label", kernel_factory, options_factory)
# ---------------------------------------------------------------------------


def _make_docker_pair(state_root, store, dataset, sif, *, accept_degraded):
    kernel, executor = _docker_kernel(
        state_root, store, accept_degraded=accept_degraded
    )
    return kernel, executor.engine if hasattr(executor, "engine") else None, executor


def _make_slurm_pair(state_root, store, dataset, sif, *, accept_degraded):
    kernel, executor = _slurm_kernel(state_root, store, accept_degraded=accept_degraded)
    return kernel, executor.engine, executor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    root = tmp_path / "state"
    root.mkdir()
    return root


@pytest.fixture
def store(tmp_path: Path) -> StoreLayout:
    return StoreLayout(root=tmp_path / "store").ensure()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    p = tmp_path / "data.h5mu"
    p.write_bytes(b"placeholder")
    return p


@pytest.fixture
def sif(tmp_path: Path) -> Path:
    p = tmp_path / "model.sif"
    p.write_bytes(b"sif-bytes")
    return p


# ---------------------------------------------------------------------------
# Test 1: default refusal — unverified_local → FAILED, engine not touched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("executor_kind", ["docker", "slurm"])
def test_default_refuses_unverified_local(
    state_root: Path,
    store: StoreLayout,
    dataset: Path,
    sif: Path,
    executor_kind: str,
) -> None:
    """Default ``accept_degraded=False`` must fail the run before engine dispatch."""
    if executor_kind == "docker":
        kernel, executor = _docker_kernel(state_root, store, accept_degraded=False)
        opts = _docker_opts(dataset=dataset, image_digest=None)  # → unverified_local
        submit_count = lambda: len(executor.supervisor.engine.containers)  # type: ignore[attr-defined]
    else:
        kernel, executor = _slurm_kernel(state_root, store, accept_degraded=False)
        opts = _slurm_opts(
            dataset=dataset, sif=sif, image_digest=None
        )  # → unverified_local
        submit_count = lambda: executor.engine.submit_count  # type: ignore[attr-defined]

    async def _run() -> dict:
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_run())

    assert snap["primary_state"] == PrimaryState.FAILED.value, snap
    assert "refused to launch" in (snap.get("failure_reason") or ""), snap
    assert "unverified_local" in (snap.get("failure_reason") or ""), snap
    # The engine must NOT have received a job.
    assert submit_count() == 0


# ---------------------------------------------------------------------------
# Test 2: opt-in — accept_degraded=True admits unverified_local
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("executor_kind", ["docker", "slurm"])
def test_accept_degraded_true_admits_unverified_local(
    state_root: Path,
    store: StoreLayout,
    dataset: Path,
    sif: Path,
    executor_kind: str,
) -> None:
    """``accept_degraded=True`` lets an unverified_local run proceed to ARTIFACT_SUCCESS."""
    if executor_kind == "docker":
        kernel, executor = _docker_kernel(state_root, store, accept_degraded=True)
        opts = _docker_opts(dataset=dataset, image_digest=None)
    else:
        kernel, executor = _slurm_kernel(state_root, store, accept_degraded=True)
        opts = _slurm_opts(dataset=dataset, sif=sif, image_digest=None)

    async def _run() -> dict:
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_run())
    assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value, snap


# ---------------------------------------------------------------------------
# Test 3: strict-acceptable identity (registry_digest) is always admitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("executor_kind", ["docker", "slurm"])
def test_registry_digest_always_admitted(
    state_root: Path,
    store: StoreLayout,
    dataset: Path,
    sif: Path,
    executor_kind: str,
) -> None:
    """A ``registry_digest`` identity passes even when ``accept_degraded=False``."""
    digest = "sha256:" + "b" * 64
    if executor_kind == "docker":
        kernel, executor = _docker_kernel(state_root, store, accept_degraded=False)
        opts = _docker_opts(dataset=dataset, image_digest=digest)
    else:
        kernel, executor = _slurm_kernel(state_root, store, accept_degraded=False)
        opts = _slurm_opts(dataset=dataset, sif=sif, image_digest=digest)

    async def _run() -> dict:
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_run())
    assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value, snap


# ---------------------------------------------------------------------------
# Test 4: broker is released after the G1 refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("executor_kind", ["docker", "slurm"])
def test_broker_released_after_refusal(
    state_root: Path,
    store: StoreLayout,
    dataset: Path,
    sif: Path,
    executor_kind: str,
) -> None:
    """The admission slot must be released even when G1 aborts before launch."""
    if executor_kind == "docker":
        kernel, executor = _docker_kernel(state_root, store, accept_degraded=False)
        opts = _docker_opts(dataset=dataset, image_digest=None)
        broker = executor.broker
    else:
        kernel, executor = _slurm_kernel(state_root, store, accept_degraded=False)
        opts = _slurm_opts(dataset=dataset, sif=sif, image_digest=None)
        broker = executor.broker

    async def _run() -> None:
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel.shutdown()

    asyncio.run(_run())
    # After the aborted run the ledger must not have any active reservations.
    assert len(broker.ledger.by_attempt) == 0
