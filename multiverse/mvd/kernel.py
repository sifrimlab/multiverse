"""The mvd kernel (STRATEGY R1 / R2 / R6).

Single asyncio process. Hot path only: journal, lease manager, Docker
supervisor, state machine, promotion driver, cancellation driver, and the
seven-verb API. Plugins (MLflow sync, GC, doctor, exporter, registration)
talk to this kernel through the socket transport and the artifact
filesystem; they do not share writable state with the kernel.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from ..artifact import (BootContext, compute_params_hash,
                        new_physical_attempt_id)
from ..broker import ResourceBroker, reconstruct_ledger_from_journal
from ..journal import JournalKind, JournalLayout, JournalReader, JournalWriter
from ..state_paths import resolve_user_id
from .api import KERNEL_VERBS, KernelAPI
from .events import EventKind, KernelEvent
from .executor import NullRunExecutor, RunExecutor
from .runs import (RunRecord, RunRegistry, assert_projection_status_valid,
                   new_run_record)
from .state import PrimaryState, assert_valid_transition


@dataclass
class KernelConfig:
    """Knobs the kernel cares about at boot."""

    state_root: Path
    mvd_version: str = "0.1.0-mvd"
    git_commit: Optional[str] = None
    accept_degraded: bool = False
    paused: bool = False
    """If True, the kernel refuses new ``submit_run`` calls until
    ``resume()`` is invoked. Used by ``mvd-rebuild-index`` (Milestone 8)
    as the paused-kernel maintenance lock (R1)."""
    user_id: str = field(default_factory=resolve_user_id)
    """Owner of the runs this kernel produces. Informational under the
    current single-user product boundary (STRATEGY M1); reserved so that
    multi-user-future does not require retrofitting tenancy into journal
    records, broker keys, and projection plugin paths. Override via
    ``$MVEXP_USER_ID`` or pass explicitly."""


class Kernel:  # implements KernelAPI
    """The seven-verb kernel."""

    def __init__(
        self,
        config: KernelConfig,
        *,
        executor: Optional[RunExecutor] = None,
        journal: Optional[JournalWriter] = None,
        boot: Optional[BootContext] = None,
        broker: Optional[ResourceBroker] = None,
    ) -> None:
        self._config = config
        self._boot = boot or BootContext.new(
            mvd_version=config.mvd_version,
            git_commit=config.git_commit,
        )

        layout = JournalLayout.at(config.state_root / "journal").ensure()
        self._journal = journal or JournalWriter(layout, boot_id=self._boot.boot_id)
        self._executor: RunExecutor = executor or NullRunExecutor()
        # STRATEGY M3: if a broker is provided, ``replay_from_journal`` will
        # reconstruct its reservation ledger from the journal. The kernel
        # also synthesizes ``RESERVATION_RELEASED`` records for any
        # reservations stranded by a crash (granted, but the run has since
        # reached a terminal state — see :meth:`replay_from_journal`).
        self._broker: Optional[ResourceBroker] = broker

        self._registry = RunRegistry()
        self._idempotency_index: Dict[str, str] = {}
        self._event_subscribers: Dict[str, List[asyncio.Queue[KernelEvent]]] = {}
        self._execution_tasks: Dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # boot / shutdown
    # ------------------------------------------------------------------

    def boot_id(self) -> str:
        return self._boot.boot_id

    def replay_from_journal(self) -> None:
        """Rebuild the in-memory registry from the journal.

        Called at kernel construction time by callers that boot against an
        existing journal. Idempotent. Does not start any execution tasks; it
        restores the durable facts needed by the GUI/index projection so a
        restarted local controller can still list prior attempts.
        """
        reader = JournalReader(JournalLayout.at(self._config.state_root / "journal"))
        self._registry = RunRegistry()
        self._idempotency_index.clear()
        all_records = list(reader.replay().records)
        for record in all_records:
            attempt = record.physical_attempt_id
            if record.kind is JournalKind.JOB_INTENT and attempt:
                run_record = new_run_record(
                    physical_attempt_id=attempt,
                    manifest_path=record.payload.get("manifest_path"),
                    logical_run_id=record.logical_run_id,
                    options=record.payload.get("options"),
                )
                run_record.submitted_wall_iso = record.wall_iso
                idempotency_key = run_record.options.get("idempotency_key")
                if idempotency_key:
                    self._idempotency_index[str(idempotency_key)] = attempt
                self._registry.add(run_record)
                continue

            if not attempt or not self._registry.has(attempt):
                continue

            run_record = self._registry.get(attempt)
            if record.logical_run_id and not run_record.logical_run_id:
                run_record.logical_run_id = record.logical_run_id

            if record.kind is JournalKind.STATE_TRANSITION:
                next_state = record.payload.get("to_state")
                if next_state:
                    try:
                        run_record.primary_state = PrimaryState(next_state)
                    except ValueError:
                        continue
                reason = record.payload.get("reason")
                if reason:
                    run_record.failure_reason = str(reason)
            elif record.kind is JournalKind.CONTAINER_LAUNCH:
                labels = record.payload.get("labels") or {}
                workspace = labels.get("multiverse.workspace")
                if workspace:
                    run_record.workspace_dir = str(workspace)
            elif record.kind is JournalKind.PROMOTE_PREPARE:
                run_record.workspace_dir = record.payload.get("workspace_dir")
                run_record.artifact_dir = record.payload.get("final_artifact_dir")
            elif record.kind is JournalKind.PROMOTE_COMMIT_MANIFEST:
                run_record.artifact_dir = record.payload.get(
                    "artifact_dir", run_record.artifact_dir
                )
            elif record.kind is JournalKind.PROMOTION_QUARANTINE:
                source = record.payload.get("source")
                if source and not run_record.failure_reason:
                    run_record.failure_reason = f"promotion quarantined: {source}"
            elif record.kind is JournalKind.CANCEL_REQUESTED:
                run_record.cancel_requested = True
            elif record.kind is JournalKind.CANCELLED:
                run_record.primary_state = PrimaryState.CANCELLED
                run_record.cancel_requested = True
            elif record.kind is JournalKind.PROJECTION_STATUS:
                plugin = record.payload.get("plugin")
                status = record.payload.get("status")
                if plugin and status:
                    run_record.projections[plugin] = status

        # STRATEGY M3: reconstruct the broker's reservation ledger from
        # the journal. Any reservation that is granted but whose run is
        # now in a terminal state was stranded by a crash — synthesize a
        # RESERVATION_RELEASED so the ledger never lies about live
        # reservations. The release is durable: future replays see the
        # ledger as empty.
        if self._broker is not None:
            ledger = reconstruct_ledger_from_journal(all_records)
            self._broker.ledger = ledger
            for attempt in list(ledger.by_attempt):
                if not self._registry.has(attempt):
                    self._broker.release(attempt, reason="crash_recovery_unknown_run")
                    continue
                state = self._registry.get(attempt).primary_state
                if state.is_terminal:
                    self._broker.release(attempt, reason="crash_recovery")

    async def shutdown(self) -> None:
        for task in self._execution_tasks.values():
            task.cancel()
        for task in self._execution_tasks.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._execution_tasks.clear()
        self._journal.close()

    # ------------------------------------------------------------------
    # internal helpers used by the executor
    # ------------------------------------------------------------------

    async def transition(
        self,
        physical_attempt_id: str,
        *,
        to_state: PrimaryState,
        reason: Optional[str] = None,
    ) -> None:
        record = self._registry.get(physical_attempt_id)
        from_state = record.primary_state
        assert_valid_transition(from_state, to_state)
        self._journal.append(
            JournalKind.STATE_TRANSITION,
            payload={
                "from_state": from_state.value,
                "to_state": to_state.value,
                "reason": reason,
            },
            physical_attempt_id=physical_attempt_id,
            logical_run_id=record.logical_run_id,
            prev_state=from_state.value,
            next_state=to_state.value,
        )
        seqs = self._journal.commit()
        record.primary_state = to_state
        if reason and to_state in (
            PrimaryState.FAILED,
            PrimaryState.PROMOTION_FAILED,
            PrimaryState.EVALUATION_FAILED,
        ):
            record.failure_reason = reason
        await self._broadcast(
            KernelEvent(
                kind=EventKind.STATE_TRANSITION,
                physical_attempt_id=physical_attempt_id,
                payload={
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                },
                seq=seqs[-1] if seqs else None,
            )
        )

    # ------------------------------------------------------------------
    # API: seven verbs
    # ------------------------------------------------------------------

    async def submit_run(
        self,
        *,
        manifest_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self._config.paused:
            raise RuntimeError("kernel is paused for maintenance; refuses submit_run")
        options = dict(options or {})

        # Idempotency: derive a key if the caller did not supply one.
        idempotency_key = options.get("idempotency_key")
        if idempotency_key is None:
            idempotency_key = _derive_idempotency_key(manifest_path, options)
            options["idempotency_key"] = idempotency_key
        cached = self._idempotency_index.get(idempotency_key)
        if cached is not None:
            return cached

        physical_attempt_id = new_physical_attempt_id()
        record = new_run_record(
            physical_attempt_id=physical_attempt_id,
            manifest_path=manifest_path,
            options=options,
        )
        self._journal.append(
            JournalKind.JOB_INTENT,
            payload={
                "manifest_path": manifest_path,
                "options": options,
            },
            physical_attempt_id=physical_attempt_id,
            prev_state=None,
            next_state=PrimaryState.PENDING.value,
        )
        self._journal.commit()

        self._registry.add(record)
        self._idempotency_index[idempotency_key] = physical_attempt_id

        await self._broadcast(
            KernelEvent(
                kind=EventKind.SUBMITTED,
                physical_attempt_id=physical_attempt_id,
                payload={"manifest_path": manifest_path},
            )
        )

        # Schedule execution. The executor runs as a kernel-owned task; on
        # kernel shutdown the task is cancelled.
        task = asyncio.create_task(
            self._drive_execution(record),
            name=f"mvd-exec-{physical_attempt_id}",
        )
        self._execution_tasks[physical_attempt_id] = task
        return physical_attempt_id

    async def cancel_run(self, *, physical_attempt_id: str) -> None:
        record = self._registry.get(physical_attempt_id)
        if record.primary_state.is_terminal:
            return  # idempotent — already done
        if record.cancel_requested:
            return  # already requested
        record.cancel_requested = True
        self._journal.append(
            JournalKind.CANCEL_REQUESTED,
            payload={"requested_from_state": record.primary_state.value},
            physical_attempt_id=physical_attempt_id,
            logical_run_id=record.logical_run_id,
        )
        self._journal.commit()
        await self._broadcast(
            KernelEvent(
                kind=EventKind.CANCEL_REQUESTED,
                physical_attempt_id=physical_attempt_id,
                payload={"from_state": record.primary_state.value},
            )
        )

    async def query_run(self, *, physical_attempt_id: str) -> Dict[str, Any]:
        return self._registry.get(physical_attempt_id).to_dict()

    async def list_runs(
        self,
        *,
        state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        primary = PrimaryState(state) if state else None
        records = self._registry.list(state=primary, logical_run_id=logical_run_id)
        return [r.to_dict() for r in records]

    def stream_events(
        self,
        *,
        physical_attempt_id: str,
    ) -> AsyncIterator[KernelEvent]:
        # Validate the attempt exists.
        self._registry.get(physical_attempt_id)
        queue: asyncio.Queue[KernelEvent] = asyncio.Queue()
        self._event_subscribers.setdefault(physical_attempt_id, []).append(queue)

        async def _iter() -> AsyncIterator[KernelEvent]:
            try:
                while True:
                    event = await queue.get()
                    yield event
            finally:
                subs = self._event_subscribers.get(physical_attempt_id, [])
                if queue in subs:
                    subs.remove(queue)

        return _iter()

    async def health(self) -> Dict[str, Any]:
        n = len(self._registry.records)
        active = sum(
            1
            for r in self._registry.records.values()
            if not r.primary_state.is_terminal
        )
        return {
            "ok": True,
            "boot_id": self._boot.boot_id,
            "mvd_version": self._config.mvd_version,
            "runs_total": n,
            "runs_active": active,
            "paused": self._config.paused,
            "journal_next_seq": self._journal.next_seq,
            "executor": self._executor.name,
        }

    async def report_projection_status(
        self,
        *,
        plugin: str,
        physical_attempt_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._registry.get(physical_attempt_id)
        assert_projection_status_valid(plugin, status)
        record.projections[plugin] = status
        self._journal.append(
            JournalKind.PROJECTION_STATUS,
            payload={
                "plugin": plugin,
                "status": status,
                "details": dict(details or {}),
            },
            physical_attempt_id=physical_attempt_id,
            logical_run_id=record.logical_run_id,
        )
        self._journal.commit()
        await self._broadcast(
            KernelEvent(
                kind=EventKind.PROJECTION_STATUS,
                physical_attempt_id=physical_attempt_id,
                payload={"plugin": plugin, "status": status},
            )
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _drive_execution(self, record: RunRecord) -> None:
        try:
            await self._executor.execute(record=record, kernel=self)
        except asyncio.CancelledError:
            # Kernel shutdown.
            raise
        except Exception as exc:
            reason = f"executor crashed: {type(exc).__name__}: {exc}"
            try:
                if record.primary_state is PrimaryState.PROMOTING:
                    await self.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.PROMOTION_FAILED,
                        reason=reason,
                    )
                    await self.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.RECOVERY_PENDING,
                        reason="promotion crashed; recovery required",
                    )
                elif record.primary_state is PrimaryState.EVALUATING:
                    await self.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.EVALUATION_FAILED,
                        reason=reason,
                    )
                    await self.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.RECOVERY_PENDING,
                        reason="evaluation crashed; recovery required",
                    )
                else:
                    await self.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.FAILED,
                        reason=reason,
                    )
            except ValueError:
                # Already terminal or no legal crash transition remains.
                pass

    async def _broadcast(self, event: KernelEvent) -> None:
        for queue in self._event_subscribers.get(event.physical_attempt_id, []):
            await queue.put(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_idempotency_key(manifest_path: str, options: Dict[str, Any]) -> str:
    """Stable, deterministic key derived from manifest path + canonical
    options. Two submits with the same payload and no explicit key
    collapse to the same run."""
    import json

    payload = {
        "manifest_path": manifest_path,
        "options": {k: v for k, v in options.items() if k != "idempotency_key"},
    }
    return compute_params_hash(payload)


# Re-export verb names so tests can ``from multiverse.mvd import KERNEL_VERBS``.
__all__ = ["KERNEL_VERBS", "Kernel", "KernelConfig", "KernelAPI"]
