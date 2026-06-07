"""G3 acceptance tests — rebuild-index reconstructs the reservation timeline.

Acceptance criteria (STRATEGY G3):
* A journal with grant+release produces two reservation_events rows in seq order.
* A grant with no release (kernel-crashed mid-run) produces one 'granted' row;
  verify_projection_against_journal does NOT flag it as drift.
* Full acceptance: write a journal with three attempts, delete the projection,
  run rebuild_index, query the timeline via the facade, assert count and order.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from multiverse.artifact import BootContext
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.docker_supervisor import (DockerSupervisor,
                                          InMemoryContainerEngine)
from multiverse.index.rebuilder import rebuild_index
from multiverse.index.sqlite_index import open_index
from multiverse.index_projection import (reservation_events_for,
                                         verify_projection_against_journal)
from multiverse.journal import (JournalKind, JournalLayout, JournalReader,
                                JournalWriter)
from multiverse.mvd import (Kernel, KernelConfig, MvdDockerExecutor,
                            PrimaryState, build_executor_options)
from multiverse.promotion import StoreLayout

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _producer_and_exit(engine: InMemoryContainerEngine):
    def _p(workspace: Path, _params: Any) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            f.create_dataset("latent", data=np.zeros((4, 4), dtype=np.float32))
        for c in reversed(list(engine.containers.values())):
            if not c.removed and c.state.value == "running":
                engine.simulate_natural_exit(c.container_id, exit_code=0)
                return

    return _p


def _producer_fail(engine: InMemoryContainerEngine):
    """Producer that writes nothing and exits non-zero → FAILED state."""

    def _p(workspace: Path, _params: Any) -> None:
        for c in reversed(list(engine.containers.values())):
            if not c.removed and c.state.value == "running":
                engine.simulate_natural_exit(c.container_id, exit_code=1)
                return

    return _p


def _build_kernel(
    state_root: Path,
    store: StoreLayout,
    engine: InMemoryContainerEngine,
    *,
    producer,
) -> tuple[Kernel, JournalWriter]:
    boot = BootContext.new(mvd_version="0.1.0-test")
    config = KernelConfig(state_root=state_root)
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id, user_id=config.user_id)
    supervisor = DockerSupervisor(
        engine=engine, journal=journal, mvd_version="0.1.0-test"
    )
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        ),
        journal=journal,
    )
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=broker,
        state_root=state_root,
        producer_hook=producer,
        poll_interval_seconds=0.0,
        max_poll_iterations=200,
        accept_degraded=False,
    )
    kernel = Kernel(
        config, executor=executor, journal=journal, boot=boot, broker=broker
    )
    return kernel, journal


def _opts(dataset: Path, *, digest: str = "sha256:" + "a" * 64) -> dict:
    return build_executor_options(
        model_slug="pca",
        model_image="pca:local",
        image_digest=digest,
        dataset_slug="demo",
        dataset_path=str(dataset),
        dataset_n_obs=4,
        params={"n_components": 4},
    )


# ---------------------------------------------------------------------------
# 1. grant + release → two rows in seq order
# ---------------------------------------------------------------------------


def test_grant_release_produces_two_rows(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    engine = InMemoryContainerEngine()

    kernel, _ = _build_kernel(
        state_root, store, engine, producer=_producer_and_exit(engine)
    )

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/m.yaml", options=_opts(dataset)
        )
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel.shutdown()
        return attempt

    attempt = asyncio.run(_run())

    db_path = state_root / "multiverse_state.db"
    with open_index(db_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store, truncate=True)
        events = idx.list_reservation_events(attempt)

    assert len(events) == 2, events
    assert events[0]["kind"] == "granted"
    assert events[1]["kind"] == "released"
    assert events[0]["seq"] < events[1]["seq"]


# ---------------------------------------------------------------------------
# 2. grant with no release (simulated crash) → one granted row, no drift
# ---------------------------------------------------------------------------


def test_grant_without_release_produces_one_row_no_drift(tmp_path: Path) -> None:
    """Simulate a kernel crash: write a RESERVATION_GRANTED record manually
    without the corresponding RESERVATION_RELEASED. The rebuilder should
    produce one 'granted' row; verify_projection_against_journal should not
    flag it as a consistency drift (it's legitimate in-flight state).
    """
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")

    boot = BootContext.new(mvd_version="0.1.0-test")
    config = KernelConfig(state_root=state_root)
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id)

    # Write a minimal JOB_INTENT + RESERVATION_GRANTED + STATE_TRANSITION(RUNNING)
    # without a RESERVATION_RELEASED.
    attempt_id = "test-crash-attempt"
    journal.append(
        JournalKind.JOB_INTENT,
        physical_attempt_id=attempt_id,
        payload={"manifest_path": "/m.yaml", "options": {}},
    )
    journal.append(
        JournalKind.RESERVATION_GRANTED,
        physical_attempt_id=attempt_id,
        payload={"ram_bytes": 256 * 1024 * 1024},
    )
    journal.append(
        JournalKind.STATE_TRANSITION,
        physical_attempt_id=attempt_id,
        payload={"from_state": "PENDING", "to_state": "RUNNING"},
    )
    journal.commit()

    db_path = state_root / "multiverse_state.db"
    with open_index(db_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store, truncate=True)
        events = idx.list_reservation_events(attempt_id)

    assert len(events) == 1
    assert events[0]["kind"] == "granted"
    assert events[0]["ram_bytes"] == 256 * 1024 * 1024

    # verify_projection_against_journal must not flag the granted-without-release
    # as a projection drift (the projection accurately mirrors the journal).
    report = verify_projection_against_journal(state_root)
    drift_ids = {d.physical_attempt_id for d in report.drifts}
    assert (
        attempt_id not in drift_ids
    ), f"unexpected drift for {attempt_id}: {report.drifts}"


# ---------------------------------------------------------------------------
# 3. Full acceptance: three attempts → rebuild restores timeline
# ---------------------------------------------------------------------------


def test_full_rebuild_restores_timeline_for_three_attempts(tmp_path: Path) -> None:
    """Write three attempts (success, failure, cancelled), delete the DB,
    rebuild, then verify the reservation event counts via the facade.
    """
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    engine = InMemoryContainerEngine()

    # Attempt 1: success — expect grant + release
    kernel1, _ = _build_kernel(
        state_root, store, engine, producer=_producer_and_exit(engine)
    )

    async def _run1() -> str:
        attempt = await kernel1.submit_run(
            manifest_path="/m.yaml",
            options=_opts(dataset, digest="sha256:" + "1" * 64),
        )
        task = kernel1._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel1.shutdown()
        return attempt

    attempt_success = asyncio.run(_run1())

    # Attempt 2: failure — expect grant + release
    engine2 = InMemoryContainerEngine()
    kernel2, _ = _build_kernel(
        state_root, store, engine2, producer=_producer_fail(engine2)
    )

    async def _run2() -> str:
        attempt = await kernel2.submit_run(
            manifest_path="/m.yaml",
            options=_opts(dataset, digest="sha256:" + "2" * 64),
        )
        task = kernel2._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel2.shutdown()
        return attempt

    attempt_failed = asyncio.run(_run2())

    # Attempt 3: manual journal-only grant (simulates a crash mid-run).
    boot3 = BootContext.new(mvd_version="0.1.0-test")
    layout3 = JournalLayout.at(state_root / "journal").ensure()
    journal3 = JournalWriter(layout3, boot_id=boot3.boot_id)
    attempt_crashed = "crashed-attempt"
    journal3.append(
        JournalKind.JOB_INTENT,
        physical_attempt_id=attempt_crashed,
        payload={"manifest_path": "/m.yaml", "options": {}},
    )
    journal3.append(
        JournalKind.RESERVATION_GRANTED,
        physical_attempt_id=attempt_crashed,
        payload={"ram_bytes": 512 * 1024 * 1024},
    )
    journal3.append(
        JournalKind.STATE_TRANSITION,
        physical_attempt_id=attempt_crashed,
        payload={"from_state": "PENDING", "to_state": "RUNNING"},
    )
    journal3.commit()

    # Delete the DB and rebuild from scratch.
    db_path = state_root / "multiverse_state.db"
    if db_path.exists():
        db_path.unlink()

    with open_index(db_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store, truncate=True)

    # Verify via the facade.
    success_events = reservation_events_for(
        state_root, physical_attempt_id=attempt_success
    )
    failed_events = reservation_events_for(
        state_root, physical_attempt_id=attempt_failed
    )
    crashed_events = reservation_events_for(
        state_root, physical_attempt_id=attempt_crashed
    )

    # Success and failure both got grant + release.
    assert len(success_events) == 2, success_events
    assert success_events[0]["kind"] == "granted"
    assert success_events[1]["kind"] == "released"

    assert len(failed_events) == 2, failed_events
    assert failed_events[0]["kind"] == "granted"
    assert failed_events[1]["kind"] == "released"

    # Crashed attempt: only the grant, no release.
    assert len(crashed_events) == 1, crashed_events
    assert crashed_events[0]["kind"] == "granted"
    assert crashed_events[0]["ram_bytes"] == 512 * 1024 * 1024
