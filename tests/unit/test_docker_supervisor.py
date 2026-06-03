"""Milestone-6 exit-gate tests for the Docker supervisor and cancel saga.

Coverage:
    1. Labels: every launched container carries the seven required labels.
    2. Lease ledger: open/renew/close, expiry.
    3. Reconcile on reboot:
        a. running container → reattached.
        b. exited-while-dead container → lease closed, exit code reported.
        c. docker rm out-of-band → disappeared=True.
    4. Cancel saga: full happy path with mocked engine; workspace moves to
       store/cancelled/<date>/<id>/; run_attempt_manifest written.
    5. Cancel saga fault-injection per step.
    6. Cancel saga on a no-longer-existing container is idempotent.
    7. Import-graph cleanliness: docker_supervisor does not import the
       real ``docker`` package at module level.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest

from multiverse.artifact import BootContext
from multiverse.docker_supervisor import (LABEL_HOST_PID, LABEL_LOGICAL_RUN_ID,
                                          LABEL_MANIFEST_HASH,
                                          LABEL_MVD_VERSION, LABEL_OWNER_TOKEN,
                                          LABEL_RUN_ID, LABEL_WORKSPACE,
                                          CancelOutcome, CancelSaga,
                                          CancelStep, ContainerEngine,
                                          ContainerState, DockerSupervisor,
                                          InMemoryContainerEngine, LeaseLedger,
                                          MultiverseLabels, RealDockerEngine,
                                          multiverse_labels)
from multiverse.journal import (JournalKind, JournalLayout, JournalReader,
                                JournalWriter)
from multiverse.promotion import StoreLayout

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def boot() -> BootContext:
    return BootContext.new(mvd_version="0.1.0-test")


@pytest.fixture
def engine() -> InMemoryContainerEngine:
    return InMemoryContainerEngine()


@pytest.fixture
def journal_writer(tmp_path: Path):
    layout = JournalLayout.at(tmp_path / "journal")
    writer = JournalWriter(layout, boot_id="boot-test")
    yield writer
    writer.close()


@pytest.fixture
def store(tmp_path: Path) -> StoreLayout:
    return StoreLayout(root=tmp_path / "store").ensure()


@pytest.fixture
def supervisor(
    engine: ContainerEngine, journal_writer: JournalWriter
) -> DockerSupervisor:
    return DockerSupervisor(
        engine=engine,
        journal=journal_writer,
        mvd_version="0.1.0-test",
    )


# ---------------------------------------------------------------------------
# 1. Labels
# ---------------------------------------------------------------------------


def test_launch_writes_all_seven_required_labels(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    res = supervisor.launch(
        physical_attempt_id="run-1",
        logical_run_id="L" * 64,
        manifest_hash="M" * 64,
        workspace=workspace,
        owner_token="own-1",
        image="multiverse-pca:1.0.0",
    )

    info = engine.inspect(res.container_id)
    labels = info.labels
    required = {
        LABEL_RUN_ID,
        LABEL_LOGICAL_RUN_ID,
        LABEL_MANIFEST_HASH,
        LABEL_WORKSPACE,
        LABEL_OWNER_TOKEN,
        LABEL_MVD_VERSION,
        LABEL_HOST_PID,
    }
    assert required.issubset(labels.keys())
    assert labels[LABEL_RUN_ID] == "run-1"
    assert labels[LABEL_OWNER_TOKEN] == "own-1"
    assert labels[LABEL_MVD_VERSION] == "0.1.0-test"
    assert MultiverseLabels.from_dict(labels) is not None


def test_launch_defaults_to_no_gpu(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    tmp_path: Path,
) -> None:
    """GPU is opt-in (issue #30): a launch without gpu_requested must not
    record a GPU request on the container."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    res = supervisor.launch(
        physical_attempt_id="run-nogpu",
        logical_run_id="L" * 64,
        manifest_hash="M" * 64,
        workspace=workspace,
        owner_token="own-1",
        image="multiverse-pca:1.0.0",
    )
    assert engine.containers[res.container_id].gpu_requested is False


def test_launch_threads_gpu_request_to_engine(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    tmp_path: Path,
) -> None:
    """An explicit gpu_requested=True must reach the engine (issue #30)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    res = supervisor.launch(
        physical_attempt_id="run-gpu",
        logical_run_id="L" * 64,
        manifest_hash="M" * 64,
        workspace=workspace,
        owner_token="own-1",
        image="multiverse-pca:1.0.0",
        gpu_requested=True,
    )
    assert engine.containers[res.container_id].gpu_requested is True


def test_launch_journals_intent_before_engine_call(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    journal_writer: JournalWriter,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    supervisor.launch(
        physical_attempt_id="run-2",
        logical_run_id="L",
        manifest_hash="M",
        workspace=workspace,
        owner_token="own-2",
        image="x:1",
    )
    journal_writer.close()
    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    kinds = [r.kind for r in reader.replay().records]
    assert JournalKind.CONTAINER_LAUNCH in kinds


# ---------------------------------------------------------------------------
# 2. Lease ledger
# ---------------------------------------------------------------------------


def test_lease_open_renew_close() -> None:
    ledger = LeaseLedger()
    lease = ledger.open(
        physical_attempt_id="r",
        container_id="c",
        workspace="/ws",
        owner_token="o",
        mvd_boot_id="B",
        ttl_seconds=60,
    )
    assert not lease.is_expired()
    ledger.renew("r")
    ledger.close("r")
    assert lease.closed
    assert ledger.active() == []


def test_lease_expiry_is_relative_to_last_renewal() -> None:
    ledger = LeaseLedger()
    lease = ledger.open(
        physical_attempt_id="r",
        container_id="c",
        workspace="/ws",
        owner_token="o",
        mvd_boot_id="B",
        ttl_seconds=1,
    )
    # Simulate a clock advance past the TTL.
    later_ns = lease.last_renewed_monotonic_ns + (2 * 1_000_000_000)
    assert lease.is_expired(now_monotonic_ns=later_ns)


# ---------------------------------------------------------------------------
# 3. Reconcile on reboot
# ---------------------------------------------------------------------------


def test_reconcile_reattaches_running_container(
    supervisor: DockerSupervisor, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    res = supervisor.launch(
        physical_attempt_id="r-running",
        logical_run_id="L",
        manifest_hash="M",
        workspace=ws,
        owner_token="o",
        image="i",
    )
    # Reboot: build a fresh ledger from "journal" (here we just pass the
    # existing lease as the expected set, mirroring the boot recovery
    # contract).
    report = supervisor.reconcile(expected=[res.lease])
    assert len(report.entries) == 1
    entry = report.entries[0]
    assert entry.reattached is True
    assert entry.state is ContainerState.RUNNING


def test_reconcile_detects_exit_while_dead(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    res = supervisor.launch(
        physical_attempt_id="r-exited",
        logical_run_id="L",
        manifest_hash="M",
        workspace=ws,
        owner_token="o",
        image="i",
    )
    engine.simulate_natural_exit(res.container_id, exit_code=2)

    report = supervisor.reconcile(expected=[res.lease])
    entry = report.entries[0]
    assert entry.state is ContainerState.EXITED
    assert entry.exit_code == 2
    assert entry.reattached is False
    assert res.lease.closed


def test_reconcile_detects_docker_rm_out_of_band(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    res = supervisor.launch(
        physical_attempt_id="r-rm",
        logical_run_id="L",
        manifest_hash="M",
        workspace=ws,
        owner_token="o",
        image="i",
    )
    engine.simulate_docker_rm(res.container_id)
    report = supervisor.reconcile(expected=[res.lease])
    entry = report.entries[0]
    assert entry.disappeared is True


def test_reconcile_one_polling(
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    res = supervisor.launch(
        physical_attempt_id="r-poll",
        logical_run_id="L",
        manifest_hash="M",
        workspace=ws,
        owner_token="o",
        image="i",
    )
    # Running tick.
    entry = supervisor.reconcile_one(res.lease)
    assert entry.state is ContainerState.RUNNING

    engine.simulate_natural_exit(res.container_id, exit_code=0, oom_killed=True)
    entry = supervisor.reconcile_one(res.lease)
    assert entry.state is ContainerState.EXITED
    assert entry.oom_killed is True


# ---------------------------------------------------------------------------
# 4. Cancel saga happy path
# ---------------------------------------------------------------------------


def _launch_and_lease(
    supervisor: DockerSupervisor,
    tmp_path: Path,
    *,
    attempt: str = "r-cancel",
):
    ws = tmp_path / f"ws-{attempt}"
    ws.mkdir()
    (ws / "container.log").write_text("partial output\n", encoding="utf-8")
    res = supervisor.launch(
        physical_attempt_id=attempt,
        logical_run_id="L",
        manifest_hash="M",
        workspace=ws,
        owner_token="o",
        image="i",
    )
    return res, ws


def test_cancel_saga_happy_path(
    boot: BootContext,
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    journal_writer: JournalWriter,
    store: StoreLayout,
    tmp_path: Path,
) -> None:
    res, ws = _launch_and_lease(supervisor, tmp_path)
    saga = CancelSaga(
        engine=engine,
        journal=journal_writer,
        layout=store,
        boot=boot,
        physical_attempt_id="r-cancel",
        logical_run_id="L",
        lease=res.lease,
    )
    result = saga.run()
    assert result.outcome is CancelOutcome.CANCELLED
    assert CancelStep.CANCELLED in result.committed_steps
    # Workspace was moved.
    assert not ws.exists()
    # Cancelled dir contains the workspace contents + an attempt manifest.
    assert result.cancelled_dir is not None
    assert (result.cancelled_dir / "container.log").is_file()
    attempt = json.loads(
        (result.cancelled_dir / "run_attempt_manifest.json").read_text()
    )
    assert attempt["final_state"] == "CANCELLED"
    assert attempt["recovery_hint"]
    assert res.lease.closed


def test_cancel_saga_journals_all_steps(
    boot: BootContext,
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    journal_writer: JournalWriter,
    store: StoreLayout,
    tmp_path: Path,
) -> None:
    res, _ = _launch_and_lease(supervisor, tmp_path, attempt="r-jrn")
    saga = CancelSaga(
        engine=engine,
        journal=journal_writer,
        layout=store,
        boot=boot,
        physical_attempt_id="r-jrn",
        logical_run_id="L",
        lease=res.lease,
    )
    saga.run()
    journal_writer.close()

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    kinds = {r.kind for r in reader.replay().records}
    assert JournalKind.CANCEL_REQUESTED in kinds
    assert JournalKind.CANCEL_STOPPED in kinds
    assert JournalKind.CANCEL_KILLED in kinds
    assert JournalKind.CANCELLED in kinds


# ---------------------------------------------------------------------------
# 5. Cancel saga fault-injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kill_after",
    [
        CancelStep.REQUESTED,
        CancelStep.STOPPED,
        CancelStep.KILLED,
    ],
)
def test_cancel_saga_fault_then_rerun_is_idempotent(
    kill_after: CancelStep,
    boot: BootContext,
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    journal_writer: JournalWriter,
    store: StoreLayout,
    tmp_path: Path,
) -> None:
    res, ws = _launch_and_lease(supervisor, tmp_path, attempt=f"r-{kill_after.value}")

    class _Abort(Exception):
        pass

    def _hook(step):
        if step is kill_after:
            raise _Abort

    saga = CancelSaga(
        engine=engine,
        journal=journal_writer,
        layout=store,
        boot=boot,
        physical_attempt_id=res.labels.run_id,
        logical_run_id="L",
        lease=res.lease,
        after_step_hook=_hook,
    )
    with pytest.raises(_Abort):
        saga.run()

    # Re-running the saga is idempotent — every step accepts an already-
    # exited/already-moved condition.
    saga2 = CancelSaga(
        engine=engine,
        journal=journal_writer,
        layout=store,
        boot=boot,
        physical_attempt_id=res.labels.run_id,
        logical_run_id="L",
        lease=res.lease,
    )
    result = saga2.run()
    assert result.outcome is CancelOutcome.CANCELLED
    assert result.cancelled_dir is not None
    assert (result.cancelled_dir / "container.log").is_file()


# ---------------------------------------------------------------------------
# 6. Cancel on a container the engine no longer knows about
# ---------------------------------------------------------------------------


def test_cancel_when_container_removed_out_of_band(
    boot: BootContext,
    supervisor: DockerSupervisor,
    engine: InMemoryContainerEngine,
    journal_writer: JournalWriter,
    store: StoreLayout,
    tmp_path: Path,
) -> None:
    res, _ = _launch_and_lease(supervisor, tmp_path, attempt="r-gone")
    engine.simulate_docker_rm(res.container_id)

    saga = CancelSaga(
        engine=engine,
        journal=journal_writer,
        layout=store,
        boot=boot,
        physical_attempt_id="r-gone",
        logical_run_id="L",
        lease=res.lease,
    )
    result = saga.run()
    # Cancel still drives to CANCELLED; stopped_ok and killed_ok in the
    # journal record False but the saga does not bail out.
    assert result.outcome is CancelOutcome.CANCELLED
    journal_writer.close()
    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    records = reader.replay().records
    by_kind = {r.kind: r.payload for r in records}
    assert by_kind[JournalKind.CANCEL_STOPPED]["stopped_ok"] is False
    assert by_kind[JournalKind.CANCEL_KILLED]["killed_ok"] is False


# ---------------------------------------------------------------------------
# 7. Real Docker adapter behaviour with an injected fake client
# ---------------------------------------------------------------------------


class _FakeDockerContainer:
    def __init__(self, container_id="cid-1", *, labels=None, status="created"):
        self.id = container_id
        self.status = status
        self.started = False
        self.stopped = False
        self.killed = False
        self.removed = False
        self.attrs = {
            "Id": container_id,
            "Config": {"Labels": dict(labels or {}), "Image": "img:1"},
            "State": {
                "Status": status,
                "ExitCode": None,
                "OOMKilled": False,
                "StartedAt": "",
                "FinishedAt": "",
            },
        }

    def start(self):
        self.started = True
        self.status = "running"
        self.attrs["State"]["Status"] = "running"

    def reload(self):
        pass

    def stop(self, timeout=10):
        self.stopped = True
        self.attrs["State"]["Status"] = "exited"
        self.attrs["State"]["ExitCode"] = 0

    def kill(self):
        self.killed = True
        self.attrs["State"]["Status"] = "exited"
        self.attrs["State"]["ExitCode"] = 137

    def remove(self, force=False):
        self.removed = True


class _FakeContainers:
    def __init__(self):
        self.created_kwargs = None
        self.container = None

    def create(self, **kwargs):
        self.created_kwargs = kwargs
        self.container = _FakeDockerContainer(labels=kwargs.get("labels"))
        return self.container

    def get(self, container_id):
        if self.container is None or self.container.id != container_id:
            raise KeyError(container_id)
        return self.container

    def list(self, all=False, filters=None):
        if self.container is None:
            return []
        labels = dict(self.container.attrs["Config"]["Labels"])
        wanted = filters.get("label", []) if filters else []
        for item in wanted:
            k, _, v = item.partition("=")
            if labels.get(k) != v:
                return []
        return [self.container]


class _FakeDockerClient:
    def __init__(self):
        self.containers = _FakeContainers()

    def ping(self):
        return True


def test_real_docker_engine_launches_with_expanded_mounts() -> None:
    client = _FakeDockerClient()
    engine = RealDockerEngine(client=client)
    info = engine.launch(
        image="img:1",
        command=["run"],
        labels={LABEL_RUN_ID: "r1"},
        env={"A": "B"},
        volumes={"/host/data.h5mu": "/input/data.h5mu", "/host/ws": "/output"},
        mem_limit="1g",
        name="mvd-r1",
    )

    assert info.state is ContainerState.RUNNING
    assert info.labels[LABEL_RUN_ID] == "r1"
    kwargs = client.containers.created_kwargs
    assert kwargs["volumes"]["/host/data.h5mu"] == {
        "bind": "/input/data.h5mu",
        "mode": "ro",
    }
    assert kwargs["volumes"]["/host/ws"] == {"bind": "/output", "mode": "rw"}
    assert kwargs["mem_limit"] == "1g"


def test_real_docker_engine_control_methods_delegate_to_client() -> None:
    client = _FakeDockerClient()
    engine = RealDockerEngine(client=client)
    info = engine.launch(
        image="img:1",
        labels={LABEL_RUN_ID: "r2"},
        command=None,
        env=None,
        volumes=None,
        mem_limit=None,
        name=None,
    )

    listed = engine.list_by_labels(labels={LABEL_RUN_ID: "r2"})
    assert [c.container_id for c in listed] == [info.container_id]

    engine.stop(info.container_id, timeout=1)
    assert client.containers.container.stopped is True
    assert engine.inspect(info.container_id).state is ContainerState.EXITED

    engine.kill(info.container_id)
    assert client.containers.container.killed is True

    engine.remove(info.container_id, force=True)
    assert client.containers.container.removed is True


# ---------------------------------------------------------------------------
# 8. Import-graph cleanliness
# ---------------------------------------------------------------------------


def test_docker_supervisor_does_not_pull_in_docker_sdk() -> None:
    """Run the check in a subprocess so the parent's ``sys.modules``
    stays intact; otherwise a later test in the suite would see stale
    bindings to a removed ``docker`` module (which broke
    ``test_event_stream`` historically)."""
    import subprocess

    script = (
        "import sys\n"
        "import multiverse.docker_supervisor  # noqa: F401\n"
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
        "docker_supervisor leaked SDK import: "
        f"{result.stdout.strip()!r}\nstderr: {result.stderr}"
    )


def test_no_real_docker_imports_in_kernel_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel in (
        "multiverse/docker_supervisor/labels.py",
        "multiverse/docker_supervisor/leases.py",
        "multiverse/docker_supervisor/supervisor.py",
        "multiverse/docker_supervisor/cancel_saga.py",
        "multiverse/docker_supervisor/client.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert not re.search(
            r"^\s*import\s+docker\b", text, re.MULTILINE
        ), f"top-level `import docker` found in {rel}"
