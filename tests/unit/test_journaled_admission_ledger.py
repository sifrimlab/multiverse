"""STRATEGY M3 acceptance: the broker's reservation ledger is durable.

Granting a reservation appends ``RESERVATION_GRANTED`` to the journal
*before* the in-memory ledger is mutated; release appends
``RESERVATION_RELEASED``. On crash recovery the ledger is rebuilt from
the journal alone — Docker labels are no longer load-bearing, which is
the precondition for running under Apptainer (no labels).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pytest

from multiverse.apptainer import InMemoryApptainerEngine
from multiverse.artifact import BootContext
from multiverse.broker import (
    HostMetrics,
    InMemoryHostObserver,
    ReservationLedger,
    ResourceBroker,
    ResourceRequest,
    reconstruct_ledger_from_journal,
)
from multiverse.docker_supervisor import (
    DockerSupervisor,
    InMemoryContainerEngine,
)
from multiverse.journal import (
    JournalKind,
    JournalLayout,
    JournalReader,
    JournalWriter,
)
from multiverse.mvd import (
    Kernel,
    KernelConfig,
    MvdDockerExecutor,
    build_executor_options,
)
from multiverse.promotion import StoreLayout


pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# Low-level: broker writes journal records on grant/release
# ---------------------------------------------------------------------------


def _writer(tmp_path: Path) -> JournalWriter:
    layout = JournalLayout.at(tmp_path / "journal").ensure()
    return JournalWriter(layout, boot_id="boot-test")


def _replay(tmp_path: Path):
    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    return reader.replay().records


def _metrics(ram_free: int = 8 * 1024**3) -> HostMetrics:
    return HostMetrics(ram_free_bytes=ram_free, ram_total_bytes=16 * 1024**3)


def test_admit_writes_reservation_granted(tmp_path: Path) -> None:
    journal = _writer(tmp_path)
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    request = ResourceRequest(ram_bytes=1024 * 1024 * 512)
    decision = broker.admit(physical_attempt_id="r1", request=request)
    assert decision.admitted
    journal.close()

    records = _replay(tmp_path)
    grants = [r for r in records if r.kind is JournalKind.RESERVATION_GRANTED]
    assert len(grants) == 1
    assert grants[0].physical_attempt_id == "r1"
    payload = grants[0].payload
    assert int(payload["ram_bytes"]) == request.ram_bytes
    assert int(payload["vram_bytes"]) == 0
    assert payload["gpu_index"] is None


def test_failed_admission_does_not_write_grant(tmp_path: Path) -> None:
    journal = _writer(tmp_path)
    # Not enough RAM to satisfy the request.
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics(ram_free=1)), journal=journal
    )
    decision = broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=1024 * 1024 * 1024),
    )
    assert not decision.admitted
    journal.close()

    records = _replay(tmp_path)
    assert not any(r.kind is JournalKind.RESERVATION_GRANTED for r in records)


def test_release_writes_reservation_released(tmp_path: Path) -> None:
    journal = _writer(tmp_path)
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=1024),
    )
    broker.release("r1", reason="terminal")
    journal.close()

    records = _replay(tmp_path)
    kinds = [r.kind for r in records]
    assert kinds.count(JournalKind.RESERVATION_GRANTED) == 1
    assert kinds.count(JournalKind.RESERVATION_RELEASED) == 1


def test_release_is_idempotent_on_journal(tmp_path: Path) -> None:
    """Calling release twice (once from classify_exit, once from the
    executor's finally) must not produce two RELEASE records — the
    second call is a no-op because the in-memory ledger is empty."""
    journal = _writer(tmp_path)
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=1024),
    )
    broker.release("r1")
    broker.release("r1")
    journal.close()

    records = _replay(tmp_path)
    assert (
        sum(1 for r in records if r.kind is JournalKind.RESERVATION_RELEASED) == 1
    )


# ---------------------------------------------------------------------------
# reconstruct_ledger_from_journal
# ---------------------------------------------------------------------------


def test_reconstruct_picks_up_in_flight_grants(tmp_path: Path) -> None:
    journal = _writer(tmp_path)
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=1234),
    )
    broker.admit(
        physical_attempt_id="r2",
        request=ResourceRequest(ram_bytes=5678),
    )
    broker.release("r1")
    journal.close()

    ledger = reconstruct_ledger_from_journal(_replay(tmp_path))
    assert set(ledger.by_attempt) == {"r2"}
    assert ledger.by_attempt["r2"].ram_bytes == 5678


def test_reconstruct_empty_when_all_released(tmp_path: Path) -> None:
    journal = _writer(tmp_path)
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    broker.admit(physical_attempt_id="r1", request=ResourceRequest(ram_bytes=1))
    broker.release("r1")
    journal.close()

    ledger = reconstruct_ledger_from_journal(_replay(tmp_path))
    assert ledger.by_attempt == {}


# ---------------------------------------------------------------------------
# Crash-recovery: kernel rehydrates the ledger from the journal alone
# ---------------------------------------------------------------------------


def _good_producer(n_obs: int):
    def _producer(workspace: Path, params: Mapping[str, Any]) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            f.create_dataset(
                "latent",
                data=np.zeros((n_obs, 4), dtype=np.float32),
            )
    return _producer


def _producer_then_exit_first_container(engine, workspace: Path, *, n_obs: int) -> None:
    _good_producer(n_obs)(workspace, {})
    for c in reversed(list(engine.containers.values())):
        if c.removed:
            continue
        if c.state.value == "running":
            engine.simulate_natural_exit(c.container_id, exit_code=0)
            return


def _build_executor_kernel(state_root: Path, engine, dataset_file: Path):
    boot = BootContext.new(mvd_version="0.1.0-test")
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id)
    store = StoreLayout(root=state_root / "store").ensure()
    supervisor = DockerSupervisor(engine=engine, journal=journal, mvd_version="0.1.0-test")
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()),
        journal=journal,
    )
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=broker,
        state_root=state_root,
        producer_hook=lambda ws, _: _producer_then_exit_first_container(
            engine, ws, n_obs=4
        ),
        poll_interval_seconds=0.0,
        max_poll_iterations=200,
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=journal,
        boot=boot,
        broker=broker,
    )
    return kernel, broker, journal


def _opts(dataset_file: Path):
    return build_executor_options(
        model_slug="pca",
        model_image="multiverse-pca:1.0.0",
        image_digest="sha256:" + "a" * 64,
        dataset_slug="demo",
        dataset_path=str(dataset_file),
        dataset_n_obs=4,
        dataset_n_vars=8,
        params={"n_components": 4},
        manifest_text="schema_version: '1'\n",
    )


@pytest.mark.parametrize(
    "engine_factory",
    [InMemoryContainerEngine, InMemoryApptainerEngine],
    ids=["docker", "apptainer"],
)
def test_successful_run_releases_durably(tmp_path: Path, engine_factory) -> None:
    dataset_file = tmp_path / "data.h5mu"
    dataset_file.write_bytes(b"placeholder")
    state_root = tmp_path / "state"
    state_root.mkdir()

    engine = engine_factory()
    kernel, broker, journal = _build_executor_kernel(state_root, engine, dataset_file)

    async def _run() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/m.yaml", options=_opts(dataset_file)
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        await kernel.shutdown()
        return attempt

    attempt = asyncio.run(_run())
    # In-memory ledger is empty after a clean run.
    assert broker.ledger.by_attempt == {}
    # And the journal carries paired grant/release records.
    layout = JournalLayout.at(state_root / "journal")
    records = JournalReader(layout).replay().records
    grants = [r for r in records if r.kind is JournalKind.RESERVATION_GRANTED]
    releases = [r for r in records if r.kind is JournalKind.RESERVATION_RELEASED]
    assert len(grants) == 1 and grants[0].physical_attempt_id == attempt
    assert len(releases) == 1 and releases[0].physical_attempt_id == attempt


@pytest.mark.parametrize(
    "engine_factory",
    [InMemoryContainerEngine, InMemoryApptainerEngine],
    ids=["docker", "apptainer"],
)
def test_crash_after_grant_before_dispatch_recovers(
    tmp_path: Path, engine_factory
) -> None:
    """Simulate the worst-case race: a RESERVATION_GRANTED is durably
    persisted, then the kernel dies before the executor's finally-block
    can release. On the next boot, ``replay_from_journal`` must observe
    the in-flight grant and either reconcile it (if the run reached a
    terminal state) or carry it forward (if it's still alive)."""
    state_root = tmp_path / "state"
    state_root.mkdir()

    # Hand-write the journal: one JOB_INTENT, one RESERVATION_GRANTED,
    # then a STATE_TRANSITION to FAILED — no RESERVATION_RELEASED.
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-crash")
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml", "options": {}},
        physical_attempt_id="r-crash",
    )
    writer.append(
        JournalKind.RESERVATION_GRANTED,
        payload={"ram_bytes": 1024, "vram_bytes": 0, "gpu_index": None, "disk_bytes_per_path": {}},
        physical_attempt_id="r-crash",
    )
    writer.append(
        JournalKind.STATE_TRANSITION,
        payload={"from_state": "PENDING", "to_state": "FAILED", "reason": "kernel crashed"},
        physical_attempt_id="r-crash",
    )
    writer.commit()
    writer.close()

    # Boot a fresh kernel + broker; replay reconstructs and reconciles.
    journal = JournalWriter(layout, boot_id="boot-recover")
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        journal=journal,
        broker=broker,
    )
    kernel.replay_from_journal()

    # The recovery release was synthesized; the broker has nothing live.
    assert broker.ledger.by_attempt == {}
    # And it's durable: replaying again sees a release record.
    asyncio.run(kernel.shutdown())
    records = JournalReader(layout).replay().records
    releases = [
        r for r in records
        if r.kind is JournalKind.RESERVATION_RELEASED
        and r.physical_attempt_id == "r-crash"
    ]
    assert len(releases) == 1
    assert releases[0].payload.get("reason") == "crash_recovery"


def test_crash_with_non_terminal_run_keeps_reservation(tmp_path: Path) -> None:
    """A reservation whose run is still in a non-terminal state is *not*
    released on recovery — the kernel doesn't know whether the work
    is still happening. The reservation is preserved so the broker keeps
    counting it against admission decisions."""
    state_root = tmp_path / "state"
    state_root.mkdir()
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-x")
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml", "options": {}},
        physical_attempt_id="r-live",
    )
    writer.append(
        JournalKind.RESERVATION_GRANTED,
        payload={"ram_bytes": 4096, "vram_bytes": 0, "gpu_index": None, "disk_bytes_per_path": {}},
        physical_attempt_id="r-live",
    )
    writer.append(
        JournalKind.STATE_TRANSITION,
        payload={"from_state": "PENDING", "to_state": "RUNNING"},
        physical_attempt_id="r-live",
    )
    writer.commit()
    writer.close()

    journal = JournalWriter(layout, boot_id="boot-recover")
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()), journal=journal
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        journal=journal,
        broker=broker,
    )
    kernel.replay_from_journal()

    assert "r-live" in broker.ledger.by_attempt
    assert broker.ledger.by_attempt["r-live"].ram_bytes == 4096
    asyncio.run(kernel.shutdown())


# ---------------------------------------------------------------------------
# ReservationLedger.has
# ---------------------------------------------------------------------------


def test_ledger_has_predicate() -> None:
    ledger = ReservationLedger()
    ledger.reserve("r1", ResourceRequest(ram_bytes=1))
    assert ledger.has("r1")
    ledger.release("r1")
    assert not ledger.has("r1")
