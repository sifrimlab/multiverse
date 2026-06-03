"""Docker supervisor — launch, reconcile, classify (STRATEGY S7).

The supervisor is the kernel's sole owner of the Docker control plane:

* it tags every container with the run-identity labels (R7 / S7);
* it issues a lease per running container and refreshes it on each tick;
* it reconciles state by querying the engine *by label*, not by tracking
  Python-held container IDs.

On startup the kernel reads its own journal, queries the engine for every
container labelled ``multiverse.run_id``, and feeds the ``reconcile``
report to its state machine. ``reconcile`` does not itself mutate run
state; it returns a structured description and lets the kernel decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..journal import JournalKind, JournalWriter
from .client import ContainerEngine, ContainerInfo, ContainerState
from .errors import NoSuchContainerError
from .labels import (LABEL_RUN_ID, MultiverseLabels, label_query,
                     multiverse_labels)
from .leases import ContainerLease, LeaseLedger


@dataclass
class LaunchResult:
    container_id: str
    labels: MultiverseLabels
    lease: ContainerLease


@dataclass
class ReconcileEntry:
    physical_attempt_id: str
    container_id: str
    state: ContainerState
    exit_code: Optional[int] = None
    oom_killed: bool = False
    reattached: bool = False
    """``True`` if the kernel reattached to a still-running container after a
    crash. ``False`` if the lease was closed (container exited while we
    were dead) or the container disappeared (``docker rm``)."""
    disappeared: bool = False
    """``True`` if the container is unknown to the engine — typically
    ``docker rm`` ran out of band."""


@dataclass
class ReconcileReport:
    entries: List[ReconcileEntry] = field(default_factory=list)
    unowned_containers: List[ContainerInfo] = field(default_factory=list)
    """Containers labelled ``multiverse.*`` but not matching any lease we
    journalled — typically left over from a previous mvd boot that did not
    finish writing its lease. Surfaced to ``doctor`` for the user to
    inspect."""


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


@dataclass
class DockerSupervisor:
    """Stateless except for the lease ledger.

    All durable state lives in the journal; the ledger is rebuilt on boot
    from journal records. The supervisor is single-threaded by contract,
    matching the kernel's asyncio model (ADR §8).
    """

    engine: ContainerEngine
    journal: JournalWriter
    mvd_version: str
    ledger: LeaseLedger = field(default_factory=LeaseLedger)

    # ------------------------------------------------------------------
    # launch
    # ------------------------------------------------------------------

    def launch(
        self,
        *,
        physical_attempt_id: str,
        logical_run_id: str,
        manifest_hash: str,
        workspace: Path,
        owner_token: str,
        image: str,
        command: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        mem_limit: Optional[str] = None,
        name: Optional[str] = None,
        entrypoint: Optional[str] = None,
        gpu_requested: bool = False,
    ) -> LaunchResult:
        labels = multiverse_labels(
            run_id=physical_attempt_id,
            logical_run_id=logical_run_id,
            manifest_hash=manifest_hash,
            workspace=str(workspace),
            owner_token=owner_token,
            mvd_version=self.mvd_version,
        )

        # Journal first, then launch. If the launch fails mid-flight we
        # have an intent record we can replay against the engine — the
        # next reconcile will either find a container at that
        # physical_attempt_id label and adopt it, or notice the absence and
        # transition the run to FAILED.
        self.journal.append(
            JournalKind.CONTAINER_LAUNCH,
            payload={
                "image": image,
                "labels": labels.to_dict(),
                "mem_limit": mem_limit,
                "gpu_requested": bool(gpu_requested),
                "command": list(command or []),
                "name": name,
                "entrypoint": entrypoint,
            },
            physical_attempt_id=physical_attempt_id,
            logical_run_id=logical_run_id,
            prev_state="ADMITTED",
            next_state="RUNNING",
        )
        self.journal.commit()

        info = self.engine.launch(
            image=image,
            command=command,
            labels=labels.to_dict(),
            env=env,
            volumes=volumes,
            mem_limit=mem_limit,
            name=name,
            entrypoint=entrypoint,
            gpu_requested=gpu_requested,
        )
        lease = self.ledger.open(
            physical_attempt_id=physical_attempt_id,
            container_id=info.container_id,
            workspace=str(workspace),
            owner_token=owner_token,
            mvd_boot_id=self.journal.boot_id,
        )
        return LaunchResult(container_id=info.container_id, labels=labels, lease=lease)

    # ------------------------------------------------------------------
    # lease tick
    # ------------------------------------------------------------------

    def renew(self, physical_attempt_id: str) -> None:
        self.ledger.renew(physical_attempt_id)

    # ------------------------------------------------------------------
    # reconcile
    # ------------------------------------------------------------------

    def reconcile(
        self,
        *,
        expected: Iterable[ContainerLease],
    ) -> ReconcileReport:
        """Cross-reference the journal's intent (``expected``) against the
        engine's live state.

        ``expected`` is typically the lease ledger rebuilt from the journal
        on boot. ``reconcile`` does not write to the journal directly; the
        kernel records ``STATE_TRANSITION`` records based on the report.
        """
        report = ReconcileReport()

        # Map expected leases by physical_attempt_id for fast intersection.
        expected_map: Dict[str, ContainerLease] = {
            lease.physical_attempt_id: lease for lease in expected
        }

        # Index engine containers by run_id label for matching. The kernel
        # asks the engine for every container we expect (one label query
        # per attempt). This is O(n) in expected leases — fine for local
        # single-user workloads — and avoids relying on "list-all"
        # semantics that differ between Docker SDK versions.
        engine_index: Dict[str, ContainerInfo] = {}
        for lease in expected_map.values():
            for info in self.engine.list_by_labels(
                labels=label_query(lease.physical_attempt_id)
            ):
                run_id = info.labels.get(LABEL_RUN_ID)
                if run_id:
                    engine_index[run_id] = info

        # Walk the expected set first.
        for attempt_id, lease in expected_map.items():
            info = engine_index.pop(attempt_id, None)
            if info is None:
                # Container disappeared (docker rm, system prune).
                report.entries.append(
                    ReconcileEntry(
                        physical_attempt_id=attempt_id,
                        container_id=lease.container_id,
                        state=ContainerState.UNKNOWN,
                        disappeared=True,
                    )
                )
                continue

            if info.state is ContainerState.RUNNING:
                # Reattach.
                lease.mvd_boot_id = self.journal.boot_id
                lease.renew()
                report.entries.append(
                    ReconcileEntry(
                        physical_attempt_id=attempt_id,
                        container_id=info.container_id,
                        state=info.state,
                        reattached=True,
                    )
                )
            else:
                # Exited while we were dead.
                lease.close()
                report.entries.append(
                    ReconcileEntry(
                        physical_attempt_id=attempt_id,
                        container_id=info.container_id,
                        state=info.state,
                        exit_code=info.exit_code,
                        oom_killed=info.oom_killed,
                    )
                )

        # Anything still in engine_index is labelled multiverse.* but the
        # journal has no record of it — surface for ``doctor``.
        for info in engine_index.values():
            report.unowned_containers.append(info)

        return report

    def reconcile_one(self, lease: ContainerLease) -> ReconcileEntry:
        """Polling helper used by the kernel's per-run supervision task."""
        try:
            info = self.engine.inspect(lease.container_id)
        except NoSuchContainerError:
            return ReconcileEntry(
                physical_attempt_id=lease.physical_attempt_id,
                container_id=lease.container_id,
                state=ContainerState.UNKNOWN,
                disappeared=True,
            )

        if info.state is ContainerState.RUNNING:
            lease.renew()
            return ReconcileEntry(
                physical_attempt_id=lease.physical_attempt_id,
                container_id=info.container_id,
                state=info.state,
                reattached=False,
            )
        lease.close()
        return ReconcileEntry(
            physical_attempt_id=lease.physical_attempt_id,
            container_id=info.container_id,
            state=info.state,
            exit_code=info.exit_code,
            oom_killed=info.oom_killed,
        )
