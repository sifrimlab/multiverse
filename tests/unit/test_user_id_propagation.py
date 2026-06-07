"""G2 acceptance tests — thread user_id through journal/manifest/projection.

Acceptance criteria:
* Every journal record written with a user_id-aware JournalWriter carries that
  user_id in the serialized form.
* Pre-G2 records (no user_id in the dict) round-trip without error and read as
  None.
* The artifact manifest's produced_by.user_id matches the configured value.
* rebuild_index populates the user_id column in the projection; pre-G2 records
  leave it NULL.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from multiverse.artifact import BootContext
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.docker_supervisor import (DockerSupervisor,
                                          InMemoryContainerEngine)
from multiverse.index.rebuilder import rebuild_index
from multiverse.index.sqlite_index import open_index
from multiverse.journal import (JournalKind, JournalLayout, JournalReader,
                                JournalWriter)
from multiverse.journal.record import JournalRecord
from multiverse.mvd import (Kernel, KernelConfig, MvdDockerExecutor,
                            PrimaryState, build_executor_options)
from multiverse.promotion import StoreLayout

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _producer_and_exit(engine: InMemoryContainerEngine):
    def _p(workspace: Path, _params) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            f.create_dataset("latent", data=np.zeros((4, 4), dtype=np.float32))
        for c in reversed(list(engine.containers.values())):
            if not c.removed and c.state.value == "running":
                engine.simulate_natural_exit(c.container_id, exit_code=0)
                return

    return _p


def _build_kernel(
    state_root: Path,
    store: StoreLayout,
    engine: InMemoryContainerEngine,
    *,
    user_id: str,
) -> tuple[Kernel, MvdDockerExecutor, JournalWriter]:
    boot = BootContext.new(mvd_version="0.1.0-test")
    config = KernelConfig(state_root=state_root, user_id=user_id)
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id, user_id=config.user_id)
    supervisor = DockerSupervisor(
        engine=engine, journal=journal, mvd_version="0.1.0-test"
    )
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        )
    )
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=broker,
        state_root=state_root,
        producer_hook=_producer_and_exit(engine),
        poll_interval_seconds=0.0,
        max_poll_iterations=200,
        accept_degraded=False,
        user_id=config.user_id,
    )
    kernel = Kernel(
        config, executor=executor, journal=journal, boot=boot, broker=broker
    )
    return kernel, executor, journal


# ---------------------------------------------------------------------------
# 1. Journal records carry user_id
# ---------------------------------------------------------------------------


def test_journal_records_carry_user_id(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    engine = InMemoryContainerEngine()

    kernel, _, _ = _build_kernel(state_root, store, engine, user_id="alice")

    async def _run() -> str:
        opts = build_executor_options(
            model_slug="pca",
            model_image="pca:local",
            image_digest="sha256:" + "a" * 64,
            dataset_slug="demo",
            dataset_path=str(dataset),
            dataset_n_obs=4,
            params={"n_components": 4},
        )
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel.shutdown()
        return attempt

    asyncio.run(_run())

    layout = JournalLayout.at(state_root / "journal")
    reader = JournalReader(layout)
    replay = reader.replay()
    records_with_user = [r for r in replay.records if r.user_id is not None]
    assert len(records_with_user) > 0, "no records with user_id found"
    assert all(r.user_id == "alice" for r in records_with_user)


# ---------------------------------------------------------------------------
# 2. Pre-G2 records round-trip with user_id == None
# ---------------------------------------------------------------------------


def test_pre_g2_record_rounds_trip_without_user_id() -> None:
    """A serialized record without a user_id field loads as user_id=None."""
    d = {
        "seq": 1,
        "kind": "JOB_INTENT",
        "monotonic_ns": 123456,
        "wall_iso": "2026-01-01T00:00:00+00:00",
        "mvd_boot_id": "boot-xyz",
        "payload": {},
    }
    record = JournalRecord.from_dict(d)
    assert record.user_id is None
    # Re-serializing a None user_id produces no user_id key.
    assert "user_id" not in record.to_dict()


# ---------------------------------------------------------------------------
# 3. Artifact manifest produced_by.user_id is populated
# ---------------------------------------------------------------------------


def test_manifest_produced_by_carries_user_id(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    engine = InMemoryContainerEngine()

    kernel, _, _ = _build_kernel(state_root, store, engine, user_id="bob")

    async def _run() -> dict:
        opts = build_executor_options(
            model_slug="pca",
            model_image="pca:local",
            image_digest="sha256:" + "b" * 64,
            dataset_slug="demo",
            dataset_path=str(dataset),
            dataset_n_obs=4,
            params={"n_components": 4},
        )
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_run())
    assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value

    from multiverse.artifact import read_manifest

    manifest = read_manifest(Path(snap["artifact_dir"]))
    assert manifest.produced_by.user_id == "bob"


# ---------------------------------------------------------------------------
# 4. rebuild-index populates user_id column
# ---------------------------------------------------------------------------


def test_rebuild_index_populates_user_id(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    engine = InMemoryContainerEngine()

    kernel, _, _ = _build_kernel(state_root, store, engine, user_id="carol")

    async def _run() -> str:
        opts = build_executor_options(
            model_slug="pca",
            model_image="pca:local",
            image_digest="sha256:" + "c" * 64,
            dataset_slug="demo",
            dataset_path=str(dataset),
            dataset_n_obs=4,
            params={"n_components": 4},
        )
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel.shutdown()
        return attempt

    attempt = asyncio.run(_run())

    db_path = state_root / "multiverse_state.db"
    with open_index(db_path) as idx:
        result = rebuild_index(
            index=idx,
            state_root=state_root,
            store=store,
            truncate=True,
        )
        row = idx.get_run(attempt)

    assert row is not None
    assert row.get("user_id") == "carol"


# ---------------------------------------------------------------------------
# 5. Pre-G2 journal (no user_id) leaves column NULL
# ---------------------------------------------------------------------------


def test_rebuild_index_null_user_id_for_pre_g2_journal(tmp_path: Path) -> None:
    """A journal written without user_id produces a NULL user_id in the projection."""
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")
    engine = InMemoryContainerEngine()

    boot = BootContext.new(mvd_version="0.1.0-test")
    # Construct writer WITHOUT user_id (simulates pre-G2 writer).
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id)  # no user_id kwarg
    supervisor = DockerSupervisor(
        engine=engine, journal=journal, mvd_version="0.1.0-test"
    )
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        )
    )
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=broker,
        state_root=state_root,
        producer_hook=_producer_and_exit(engine),
        poll_interval_seconds=0.0,
        max_poll_iterations=200,
        accept_degraded=False,
        # user_id not set on executor either
    )
    config = KernelConfig(state_root=state_root)
    kernel = Kernel(
        config, executor=executor, journal=journal, boot=boot, broker=broker
    )

    async def _run() -> str:
        opts = build_executor_options(
            model_slug="pca",
            model_image="pca:local",
            image_digest="sha256:" + "d" * 64,
            dataset_slug="demo",
            dataset_path=str(dataset),
            dataset_n_obs=4,
            params={"n_components": 4},
        )
        attempt = await kernel.submit_run(manifest_path="/m.yaml", options=opts)
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task:
            await task
        await kernel.shutdown()
        return attempt

    attempt = asyncio.run(_run())

    db_path = state_root / "multiverse_state.db"
    with open_index(db_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store, truncate=True)
        row = idx.get_run(attempt)

    assert row is not None
    assert row.get("user_id") is None
