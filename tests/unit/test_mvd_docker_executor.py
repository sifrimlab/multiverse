"""Move-4 exit-gate tests for ``MvdDockerExecutor``.

Strategy v2 §4 acceptance: the CLI can submit a real Docker-backed run
through ``mvd`` and reach ``ARTIFACT_SUCCESS`` with no legacy runner
involvement.

These tests use ``InMemoryContainerEngine`` plus a ``producer`` callable
that synthesizes embeddings.h5 in the workspace so the validators pass.
The strategy points covered:

    1. Happy path: PENDING → ADMITTED → RUNNING → TRAINING_SUCCEEDED →
       EVALUATING → PROMOTING → ARTIFACT_SUCCESS, and the kernel records
       a TRACKING_PENDING projection status after success (S13 / R6).
    2. Admission failure: broker refuses → run transitions to FAILED
       without ever touching Docker.
    3. Container non-zero exit → FAILED with the right reason.
    4. OOM-killed exit → FAILED with OOM_KILLED reason classification.
    5. Validator refusal (wrong n_obs) → PROMOTION_FAILED then
       RECOVERY_PENDING with a quarantine tombstone left behind.
    6. Cancel mid-run → CANCEL_REQUESTED → CANCELLED with the cancel
       saga having run.
    7. Import-graph: the executor module does NOT import the docker SDK.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pytest

from multiverse.artifact import BootContext
from multiverse.broker import (
    HostMetrics,
    InMemoryHostObserver,
    PressureThresholds,
    ResourceBroker,
)
from multiverse.docker_supervisor import (
    DockerSupervisor,
    InMemoryContainerEngine,
)
from multiverse.journal import JournalLayout, JournalWriter
from multiverse.mvd import (
    Kernel,
    KernelConfig,
    MvdDockerExecutor,
    PrimaryState,
    build_executor_options,
)
from multiverse.promotion import StoreLayout


# ---------------------------------------------------------------------------
# helpers / fixtures
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
def dataset_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.h5mu"
    p.write_bytes(b"placeholder")
    return p


def _journal(state_root: Path, boot_id: str) -> JournalWriter:
    return JournalWriter(JournalLayout.at(state_root / "journal"), boot_id=boot_id)


def _broker_with_capacity() -> ResourceBroker:
    return ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        )
    )


def _broker_under_pressure() -> ResourceBroker:
    # 96% utilisation → CRITICAL.
    return ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=640 * 1024**2, ram_total_bytes=16 * 1024**3)
        )
    )


def _good_producer(n_obs: int):
    def _producer(workspace: Path, params: Mapping[str, Any]) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            f.create_dataset(
                "latent",
                data=np.random.default_rng(0).standard_normal((n_obs, 4)).astype(
                    np.float32
                ),
            )
    return _producer


def _bad_n_obs_producer(actual_n_obs: int):
    def _producer(workspace: Path, params: Mapping[str, Any]) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            f.create_dataset(
                "latent",
                data=np.zeros((actual_n_obs, 4), dtype=np.float32),
            )
    return _producer


def _executor(
    state_root: Path,
    store: StoreLayout,
    *,
    engine: InMemoryContainerEngine,
    broker: ResourceBroker,
    producer,
) -> tuple[MvdDockerExecutor, JournalWriter, BootContext]:
    boot = BootContext.new(mvd_version="0.1.0-test")
    writer = _journal(state_root, boot_id=boot.boot_id)
    supervisor = DockerSupervisor(
        engine=engine,
        journal=writer,
        mvd_version="0.1.0-test",
    )
    return (
        MvdDockerExecutor(
            journal=writer,
            boot=boot,
            store=store,
            supervisor=supervisor,
            broker=broker,
            state_root=state_root,
            producer_hook=producer,
            poll_interval_seconds=0.0,  # fast tests
            max_poll_iterations=200,
        ),
        writer,
        boot,
    )


def _kernel(state_root: Path, executor: MvdDockerExecutor) -> Kernel:
    return Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=executor.journal,
        boot=executor.boot,
    )


def _opts(dataset_file: Path, *, digest: str | None = "sha256:" + "a" * 64) -> dict:
    return build_executor_options(
        model_slug="pca",
        model_image="multiverse-pca:1.0.0",
        image_digest=digest,
        dataset_slug="demo",
        dataset_path=str(dataset_file),
        dataset_n_obs=4,
        dataset_n_vars=8,
        params={"n_components": 4},
        manifest_text="schema_version: '1'\n",
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_drives_run_to_artifact_success(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    engine = InMemoryContainerEngine()
    # The fake engine treats containers as RUNNING; the supervisor's
    # reconcile_one polls it. We need the engine to "exit" the container
    # after the producer runs. The simplest path: producer also marks
    # the container as exited.
    executor, _, _ = _executor(
        state_root,
        store,
        engine=engine,
        broker=_broker_with_capacity(),
        producer=lambda ws, _: _producer_then_exit_first_container(engine, ws, n_obs=4),
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        # Projection PENDING was emitted after artifact-success.
        assert snap["projections"]["mlflow"] == "TRACKING_PENDING"
        # Artifact dir was promoted and contains a verified manifest.
        artifact_dir = Path(snap["artifact_dir"])
        from multiverse.artifact import read_manifest

        manifest = read_manifest(artifact_dir)
        assert manifest.physical_attempt_id == attempt
        entries = {entry.name: entry for entry in manifest.artifacts}
        assert "embeddings.h5" in entries
        assert entries["embeddings.h5"].validated is True
        assert entries["embeddings.h5"].size > 0
        assert len(entries["embeddings.h5"].sha256) == 64
        assert (artifact_dir / "job_spec.json").is_file()
        await kernel.shutdown()

    asyncio.run(_run())


def _producer_then_exit_first_container(
    engine: InMemoryContainerEngine, workspace: Path, *, n_obs: int
) -> None:
    """Write embeddings and mark the most recently started container exited(0)."""
    _good_producer(n_obs)(workspace, {})
    # Find the most recently launched still-running container.
    for c in reversed(list(engine.containers.values())):
        if c.removed:
            continue
        if c.state.value == "running":
            engine.simulate_natural_exit(c.container_id, exit_code=0)
            return


# ---------------------------------------------------------------------------
# 2. Broker refusal short-circuits to FAILED
# ---------------------------------------------------------------------------


def test_broker_refusal_marks_failed_without_docker(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    engine = InMemoryContainerEngine()
    executor, _, _ = _executor(
        state_root,
        store,
        engine=engine,
        broker=_broker_under_pressure(),
        producer=lambda ws, _: None,
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.FAILED.value
        assert "pressure" in (snap["failure_reason"] or "").lower()
        assert engine.containers == {}, "no container should be launched"
        await kernel.shutdown()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3-4. Container exit handling
# ---------------------------------------------------------------------------


def test_non_zero_exit_marks_failed(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    engine = InMemoryContainerEngine()

    def _producer(workspace: Path, _: Mapping[str, Any]) -> None:
        for c in reversed(list(engine.containers.values())):
            if c.state.value == "running":
                engine.simulate_natural_exit(c.container_id, exit_code=2)
                return

    executor, _, _ = _executor(
        state_root, store, engine=engine, broker=_broker_with_capacity(), producer=_producer
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.FAILED.value
        assert "exited 2" in (snap["failure_reason"] or "")
        await kernel.shutdown()

    asyncio.run(_run())


def test_oom_kill_marks_failed_with_oom_reason(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    engine = InMemoryContainerEngine()

    def _producer(workspace: Path, _: Mapping[str, Any]) -> None:
        for c in reversed(list(engine.containers.values())):
            if c.state.value == "running":
                engine.simulate_natural_exit(
                    c.container_id, exit_code=137, oom_killed=True
                )
                return

    executor, _, _ = _executor(
        state_root, store, engine=engine, broker=_broker_with_capacity(), producer=_producer
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.FAILED.value
        assert "OOM" in (snap["failure_reason"] or "")
        await kernel.shutdown()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Validator refusal → quarantine + RECOVERY_PENDING
# ---------------------------------------------------------------------------


def test_validator_refusal_yields_recovery_pending_with_quarantine(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    engine = InMemoryContainerEngine()

    def _producer(workspace: Path, _: Mapping[str, Any]) -> None:
        # Wrong shape: n_obs=8 in embedding but contract expects 4.
        _bad_n_obs_producer(actual_n_obs=8)(workspace, {})
        for c in reversed(list(engine.containers.values())):
            if c.state.value == "running":
                engine.simulate_natural_exit(c.container_id, exit_code=0)
                return

    executor, _, _ = _executor(
        state_root,
        store,
        engine=engine,
        broker=_broker_with_capacity(),
        producer=_producer,
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.RECOVERY_PENDING.value
        # The promotion saga quarantined the prepared artifact dir.
        quarantine_root = store.quarantine
        assert quarantine_root.is_dir()
        reports = list(quarantine_root.rglob("QUARANTINE_REPORT.md"))
        assert reports, "quarantine report must be present"
        await kernel.shutdown()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Cancel mid-run → CANCEL_REQUESTED → CANCELLED
# ---------------------------------------------------------------------------


def test_cancel_during_running_drives_cancel_saga(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    engine = InMemoryContainerEngine()

    executor, _, _ = _executor(
        state_root,
        store,
        engine=engine,
        broker=_broker_with_capacity(),
        producer=lambda *_: None,  # container stays RUNNING
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await kernel.cancel_run(physical_attempt_id=attempt)
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.CANCELLED.value
        # And the cancel saga moved the workspace into store/cancelled/.
        cancelled = store.cancelled
        assert cancelled.is_dir()
        attempt_manifests = list(cancelled.rglob("run_attempt_manifest.json"))
        assert attempt_manifests, "cancel must leave a run_attempt_manifest.json"
        attempt_data = json.loads(attempt_manifests[0].read_text())
        assert attempt_data["final_state"] == "CANCELLED"
        await kernel.shutdown()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 7. Import-graph: docker_executor does NOT load the docker SDK
# ---------------------------------------------------------------------------


def test_docker_executor_does_not_load_docker_sdk() -> None:
    script = (
        "import sys\n"
        "from multiverse.mvd.docker_executor import MvdDockerExecutor\n"
        "if 'docker' in sys.modules:\n"
        "    print('docker leaked')\n"
        "    raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"docker_executor leaked: {result.stdout.strip()!r}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# 8. Per-run logging: container.log (host-captured) + orchestrator.log
# ---------------------------------------------------------------------------


def test_success_promotes_container_and_orchestrator_logs(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    """A successful run carries the host-captured container.log and the
    orchestrator.log into the promoted artifact dir."""
    engine = InMemoryContainerEngine()

    def _producer(workspace: Path, _: Mapping[str, Any]) -> None:
        _good_producer(4)(workspace, {})
        for c in reversed(list(engine.containers.values())):
            if c.state.value == "running":
                engine.simulate_logs(c.container_id, b"hello from stdout\n")
                engine.simulate_natural_exit(c.container_id, exit_code=0)
                return

    executor, _, _ = _executor(
        state_root, store, engine=engine, broker=_broker_with_capacity(), producer=_producer
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        artifact_dir = Path(snap["artifact_dir"])
        container_log = artifact_dir / "container.log"
        orchestrator_log = artifact_dir / "orchestrator.log"
        assert container_log.is_file()
        assert container_log.read_bytes() == b"hello from stdout\n"
        assert orchestrator_log.is_file()
        text = orchestrator_log.read_text(encoding="utf-8")
        assert "container launched" in text
        assert "ARTIFACT_SUCCESS" in text
        await kernel.shutdown()

    asyncio.run(_run())


def test_failed_run_keeps_logs_in_workspace(
    state_root: Path, store: StoreLayout, dataset_file: Path
) -> None:
    """A run that exits non-zero is not promoted; its container.log and
    orchestrator.log remain in the workspace for debugging."""
    engine = InMemoryContainerEngine()

    def _producer(workspace: Path, _: Mapping[str, Any]) -> None:
        for c in reversed(list(engine.containers.values())):
            if c.state.value == "running":
                engine.simulate_logs(c.container_id, b"boom: traceback\n")
                engine.simulate_natural_exit(c.container_id, exit_code=2)
                return

    executor, _, _ = _executor(
        state_root, store, engine=engine, broker=_broker_with_capacity(), producer=_producer
    )
    kernel = _kernel(state_root, executor)

    async def _run() -> None:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.FAILED.value
        workspace = store.workspaces / attempt
        assert (workspace / "container.log").read_bytes() == b"boom: traceback\n"
        orch = (workspace / "orchestrator.log").read_text(encoding="utf-8")
        assert "run failed" in orch
        await kernel.shutdown()

    asyncio.run(_run())
