"""End-to-end ``MvdSlurmExecutor`` lifecycle tests (STRATEGY M4).

These tests wire the executor against ``InMemorySlurmEngine`` plus a
``producer`` callable that synthesizes embeddings.h5 in the workspace
once the test sees the job become RUNNING. The intent is to exercise
the same scenarios the Docker executor covers:

* happy path: PENDING → ADMITTED → RUNNING → TRAINING_SUCCEEDED →
  EVALUATING → PROMOTING → ARTIFACT_SUCCESS.
* failure modes: COMPLETED with bad outputs, FAILED, OUT_OF_MEMORY,
  TIMEOUT, CANCELLED.
* M3 reservation accounting: every run grants + releases exactly once.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from multiverse.artifact import BootContext, ImageIdentityKind, read_manifest
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.journal import (JournalKind, JournalLayout, JournalReader,
                                JournalWriter)
from multiverse.mvd import (Kernel, KernelConfig, MvdSlurmExecutor,
                            PrimaryState, build_slurm_executor_options)
from multiverse.promotion import StoreLayout
from multiverse.slurm import InMemorySlurmEngine, SlurmJobState

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_producer(workspace: Path, *, n_obs: int) -> None:
    with h5py.File(workspace / "embeddings.h5", "w") as f:
        f.create_dataset("latent", data=np.zeros((n_obs, 4), dtype=np.float32))


def _broker_inflight(*, inflight: int = 8, journal=None) -> ResourceBroker:
    return ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=1, ram_total_bytes=1024)
        ),
        max_inflight_dispatches=inflight,
        journal=journal,
    )


def _build(
    *,
    state_root: Path,
    dataset_file: Path,
    n_obs: int = 4,
    transition_to: SlurmJobState = SlurmJobState.COMPLETED,
    poll_interval: float = 0.01,
) -> tuple[Kernel, InMemorySlurmEngine, ResourceBroker, JournalWriter]:
    boot = BootContext.new(mvd_version="0.1.0-test")
    journal = JournalWriter(
        JournalLayout.at(state_root / "journal").ensure(),
        boot_id=boot.boot_id,
    )
    store = StoreLayout(root=state_root / "store").ensure()
    engine = InMemorySlurmEngine()
    broker = _broker_inflight(journal=journal)
    executor = _DrivingSlurmExecutor(
        journal=journal,
        boot=boot,
        store=store,
        engine=engine,
        broker=broker,
        state_root=state_root,
        poll_interval_seconds=poll_interval,
        max_poll_iterations=200,
        n_obs=n_obs,
        transition_to=transition_to,
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=journal,
        boot=boot,
        broker=broker,
    )
    return kernel, engine, broker, journal


class _DrivingSlurmExecutor(MvdSlurmExecutor):
    """Real ``MvdSlurmExecutor`` plus a side-thread that flips the
    in-memory engine through its lifecycle on the test's behalf.

    The executor's polling loop sleeps ``poll_interval_seconds=0`` so
    we don't slow the test, but we still need *something* to drive the
    fake's transitions concurrently with the polling. A daemon thread
    watches for the most recent submission and progresses it.
    """

    def __init__(
        self,
        *args,
        n_obs: int,
        transition_to: SlurmJobState,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._n_obs = n_obs
        self._transition_to = transition_to

    def _record_dispatch(self, attempt_id: str, job_id: str) -> None:  # type: ignore[override]
        super()._record_dispatch(attempt_id, job_id)
        engine = self.engine  # capture
        assert isinstance(engine, InMemorySlurmEngine)
        # Run a background "scheduler" that simulates RUNNING → terminal
        # after a short delay. The producer drops the artifact while the
        # job is RUNNING.
        spec = engine.jobs[job_id].spec
        workspace = spec.workspace
        n_obs = self._n_obs
        target = self._transition_to

        def _drive() -> None:
            time.sleep(0.01)
            engine.simulate_running(job_id)
            if target is SlurmJobState.COMPLETED:
                _good_producer(workspace, n_obs=n_obs)
                engine.simulate_completed(job_id, exit_code=0)
            elif target is SlurmJobState.FAILED:
                engine.simulate_failed(job_id, exit_code=1, reason="user error")
            elif target is SlurmJobState.OUT_OF_MEMORY:
                engine.simulate_oom(job_id)
            elif target is SlurmJobState.TIMEOUT:
                engine.simulate_timeout(job_id)
            elif target is SlurmJobState.CANCELLED:
                engine.simulate_failed(job_id, exit_code=0, reason="cancelled by user")
            else:
                engine.simulate_failed(job_id, exit_code=99)

        threading.Thread(target=_drive, daemon=True).start()


def _opts(dataset_file: Path, sif_path: Path) -> dict:
    return build_slurm_executor_options(
        model_slug="pca",
        image_sif=str(sif_path),
        image_digest="sha256:" + "a" * 64,
        dataset_slug="demo",
        dataset_path=str(dataset_file),
        dataset_n_obs=4,
        dataset_n_vars=8,
        params={"n_components": 4},
        manifest_text="schema_version: '1'\n",
        partition="cpu-short",
        account="lab1",
        time_minutes=15,
        mem_gb=4,
        cpus_per_task=2,
    )


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    sif = tmp_path / "model.sif"
    sif.write_bytes(b"sif-bytes")
    return dataset, sif


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_runs_to_artifact_success(tmp_path: Path) -> None:
    dataset, sif = _dataset(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    kernel, engine, broker, journal = _build(
        state_root=state_root, dataset_file=dataset
    )

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset, sif)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value, snap
        return attempt

    attempt = asyncio.run(_run())

    # Reservation ledger emptied via paired grant/release records.
    asyncio.run(kernel.shutdown())
    records = JournalReader(JournalLayout.at(state_root / "journal")).replay().records
    grants = [r for r in records if r.kind is JournalKind.RESERVATION_GRANTED]
    releases = [r for r in records if r.kind is JournalKind.RESERVATION_RELEASED]
    assert len(grants) == 1
    assert len(releases) == 1
    assert grants[0].physical_attempt_id == attempt
    assert broker.ledger.by_attempt == {}

    # CONTAINER_LAUNCH carries the slurm job id (executor's dispatch pin).
    launches = [r for r in records if r.kind is JournalKind.CONTAINER_LAUNCH]
    assert len(launches) == 1
    assert launches[0].payload.get("engine") == engine.name
    assert "slurm_job_id" in launches[0].payload


def test_artifact_carries_image_identity(tmp_path: Path) -> None:
    dataset, sif = _dataset(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    kernel, _, _, _ = _build(state_root=state_root, dataset_file=dataset)

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset, sif)
        )
        await kernel._execution_tasks[attempt]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        return snap["artifact_dir"]

    artifact_dir = asyncio.run(_run())
    manifest = read_manifest(Path(artifact_dir))
    assert manifest.image_identity.value.startswith("sha256:")
    asyncio.run(kernel.shutdown())


# ---------------------------------------------------------------------------
# G4: dual-digest — runtime_image_identity is populated by InMemorySlurmEngine
# ---------------------------------------------------------------------------


def test_manifest_has_runtime_sif_digest(tmp_path: Path) -> None:
    """G4: InMemorySlurmEngine.sif_digest_for_submission provides a SIF digest
    that the executor threads into runtime_image_identity; the manifest carries
    kind==sif_digest and built_from==source image_digest (M2 dual-digest).
    """
    dataset, sif = _dataset(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    kernel, _, _, _ = _build(state_root=state_root, dataset_file=dataset)

    async def _run() -> dict:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset, sif)
        )
        await kernel._execution_tasks[attempt]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        return snap

    snap = asyncio.run(_run())
    assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
    manifest = read_manifest(Path(snap["artifact_dir"]))

    rti = manifest.runtime_image_identity
    assert (
        rti is not None
    ), "runtime_image_identity must be set when engine provides digest"
    assert rti.kind is ImageIdentityKind.SIF_DIGEST
    assert rti.value.startswith("sha256:")
    # built_from must echo the source image_digest (M2 dual-digest invariant).
    assert rti.built_from == "sha256:" + "a" * 64
    asyncio.run(kernel.shutdown())


def test_slurm_sif_digest_caching(tmp_path: Path) -> None:
    """G4: RealSlurmEngine.sif_digest_for_submission caches by (path, mtime_ns, size)
    so the same SIF is not rehashed on a second call with identical stat values.
    """
    import hashlib as _hashlib

    from multiverse.slurm.engine import RealSlurmEngine
    from multiverse.slurm.template import SlurmJobSpec

    sif = tmp_path / "model.sif"
    sif.write_bytes(b"fake-content")

    engine = RealSlurmEngine()
    spec = SlurmJobSpec(
        job_name="test",
        image_sif=sif,
        workspace=tmp_path,
        dataset_path=tmp_path / "data.h5mu",
    )

    digest1 = engine.sif_digest_for_submission(spec)
    digest2 = engine.sif_digest_for_submission(spec)

    # Both calls return the same digest.
    assert digest1 == digest2
    assert digest1 is not None
    assert digest1.startswith("sha256:")

    # Cache has exactly one entry.
    assert len(engine._sif_digest_cache) == 1

    # The digest matches compute_sif_digest directly.
    from multiverse.apptainer.images import compute_sif_digest

    assert digest1 == compute_sif_digest(sif)

    # Writing new content bumps the mtime/size → cache miss → new digest.
    sif.write_bytes(b"different-content")
    digest3 = engine.sif_digest_for_submission(spec)
    assert digest3 != digest1
    assert len(engine._sif_digest_cache) == 2


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "terminal_state",
    [
        SlurmJobState.FAILED,
        SlurmJobState.OUT_OF_MEMORY,
        SlurmJobState.TIMEOUT,
    ],
    ids=["failed", "oom", "timeout"],
)
def test_terminal_failure_states_transition_run_to_failed(
    tmp_path: Path, terminal_state: SlurmJobState
) -> None:
    dataset, sif = _dataset(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    kernel, _, broker, _ = _build(
        state_root=state_root, dataset_file=dataset, transition_to=terminal_state
    )

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset, sif)
        )
        await kernel._execution_tasks[attempt]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        return snap["primary_state"]

    final = asyncio.run(_run())
    assert final == PrimaryState.FAILED.value
    # M3 invariant: ledger is empty on terminal regardless of failure shape.
    assert broker.ledger.by_attempt == {}
    asyncio.run(kernel.shutdown())


# ---------------------------------------------------------------------------
# Admission failure
# ---------------------------------------------------------------------------


def test_admission_failure_does_not_dispatch_sbatch(tmp_path: Path) -> None:
    dataset, sif = _dataset(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    boot = BootContext.new(mvd_version="0.1.0-test")
    journal = JournalWriter(
        JournalLayout.at(state_root / "journal").ensure(),
        boot_id=boot.boot_id,
    )
    store = StoreLayout(root=state_root / "store").ensure()
    engine = InMemorySlurmEngine()
    # max_inflight=0 — every admission must refuse.
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=1, ram_total_bytes=1024)
        ),
        max_inflight_dispatches=0,
        journal=journal,
    )
    executor = MvdSlurmExecutor(
        journal=journal,
        boot=boot,
        store=store,
        engine=engine,
        broker=broker,
        state_root=state_root,
        poll_interval_seconds=0.0,
        max_poll_iterations=10,
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=journal,
        boot=boot,
        broker=broker,
    )

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset, sif)
        )
        await kernel._execution_tasks[attempt]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        return snap["primary_state"]

    final = asyncio.run(_run())
    assert final == PrimaryState.FAILED.value
    assert engine.submit_count == 0  # never reached sbatch
    asyncio.run(kernel.shutdown())


# ---------------------------------------------------------------------------
# Bad options
# ---------------------------------------------------------------------------


def test_missing_image_sif_fails_loudly(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    kernel, _, _, _ = _build(state_root=state_root, dataset_file=dataset)

    bad_options = build_slurm_executor_options(
        model_slug="pca",
        image_sif="",  # empty
        dataset_slug="demo",
        dataset_path=str(dataset),
        dataset_n_obs=4,
        params={},
    )

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=bad_options
        )
        await kernel._execution_tasks[attempt]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        return snap["primary_state"]

    final = asyncio.run(_run())
    assert final == PrimaryState.FAILED.value
    asyncio.run(kernel.shutdown())
