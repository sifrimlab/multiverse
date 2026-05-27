"""Cancellation saga (STRATEGY S8).

Cancel mirrors promotion's saga shape: every step journals its intent
*before* the side effect, and every step is idempotent. The user's cancel
button submits the intent and returns immediately; the kernel drives the
saga.

Sequence:
    1. CANCEL_REQUESTED  — journal record committed.
    2. CANCEL_STOPPED    — ``docker stop --time=<grace>``.
    3. CANCEL_KILLED     — ``docker kill`` after grace expires (idempotent
                           even if step 2 succeeded).
    4. Workspace + log snapshot moved to ``store/cancelled/<id>/``.
    5. ``CANCELLED`` terminal transition.
    6. MLflow run closed with status ``KILLED`` — delegated to projection
       plugin (Milestone 10).

The workspace move uses the same R5 rules as quarantine: rename only,
never delete. A run that was cancelled while in PROMOTING leaves the
prepared artifact dir; the cancel saga moves the *workspace*, not the
artifact dir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from ..artifact import (
    BootContext,
    ProducedBy,
    RunAttemptManifest,
    produced_at_now,
    write_run_attempt_manifest,
)
from ..artifact.checksums import fsync_path
from ..journal import JournalKind, JournalWriter
from ..promotion.layout import StoreLayout
from .client import ContainerEngine
from .errors import NoSuchContainerError
from .leases import ContainerLease


DEFAULT_CANCEL_GRACE_SECONDS = 10


class CancelStep(str, Enum):
    REQUESTED = "CANCEL_REQUESTED"
    STOPPED = "CANCEL_STOPPED"
    KILLED = "CANCEL_KILLED"
    WORKSPACE_PRESERVED = "WORKSPACE_PRESERVED"
    CANCELLED = "CANCELLED"


class CancelOutcome(str, Enum):
    CANCELLED = "CANCELLED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"
    FAILED = "FAILED"


@dataclass
class CancelResult:
    outcome: CancelOutcome
    committed_steps: List[CancelStep] = field(default_factory=list)
    cancelled_dir: Optional[Path] = None
    failure_reason: Optional[str] = None


@dataclass
class CancelSaga:
    """One-shot driver for a single physical_attempt_id."""

    engine: ContainerEngine
    journal: JournalWriter
    layout: StoreLayout
    boot: BootContext
    physical_attempt_id: str
    logical_run_id: str
    lease: ContainerLease
    grace_seconds: int = DEFAULT_CANCEL_GRACE_SECONDS
    after_step_hook: Optional[Callable[[CancelStep], None]] = None

    def run(self) -> CancelResult:
        result = CancelResult(outcome=CancelOutcome.FAILED)

        # Step 1: CANCEL_REQUESTED.
        self.journal.append(
            JournalKind.CANCEL_REQUESTED,
            payload={
                "container_id": self.lease.container_id,
                "grace_seconds": int(self.grace_seconds),
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
            prev_state="RUNNING",
            next_state="CANCEL_REQUESTED",
        )
        self.journal.commit()
        result.committed_steps.append(CancelStep.REQUESTED)
        self._maybe_fault(CancelStep.REQUESTED)

        # Step 2: docker stop. Idempotent — engines accept stop on an
        # exited container as a no-op.
        try:
            self.engine.stop(self.lease.container_id, timeout=self.grace_seconds)
            stopped_ok = True
        except NoSuchContainerError:
            stopped_ok = False
        self.journal.append(
            JournalKind.CANCEL_STOPPED,
            payload={"stopped_ok": stopped_ok},
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
        )
        self.journal.commit()
        result.committed_steps.append(CancelStep.STOPPED)
        self._maybe_fault(CancelStep.STOPPED)

        # Step 3: docker kill (idempotent after stop).
        try:
            self.engine.kill(self.lease.container_id)
            killed_ok = True
        except NoSuchContainerError:
            killed_ok = False
        self.journal.append(
            JournalKind.CANCEL_KILLED,
            payload={"killed_ok": killed_ok},
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
        )
        self.journal.commit()
        result.committed_steps.append(CancelStep.KILLED)
        self._maybe_fault(CancelStep.KILLED)

        # Step 4: preserve workspace. The workspace was the
        # multiverse.workspace label's value.
        cancelled_dir = self._preserve_workspace()
        result.cancelled_dir = cancelled_dir
        result.committed_steps.append(CancelStep.WORKSPACE_PRESERVED)
        self._maybe_fault(CancelStep.WORKSPACE_PRESERVED)

        # Step 5: terminal CANCELLED transition.
        self.journal.append(
            JournalKind.CANCELLED,
            payload={
                "cancelled_dir": str(cancelled_dir) if cancelled_dir else None,
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
            prev_state="CANCEL_REQUESTED",
            next_state="CANCELLED",
        )
        self.journal.commit()
        result.committed_steps.append(CancelStep.CANCELLED)

        self.lease.close()
        result.outcome = CancelOutcome.CANCELLED
        return result

    # ------------------------------------------------------------------

    def _preserve_workspace(self) -> Optional[Path]:
        workspace = Path(self.lease.workspace)
        if not workspace.exists():
            return None

        cancelled_root = self.layout.cancelled
        cancelled_root.mkdir(parents=True, exist_ok=True)
        # Same-day partition for easy GC bookkeeping.
        partition = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_root = cancelled_root / partition
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / self.physical_attempt_id
        # Uniquify if a previous cancel-of-same-attempt left a sibling.
        if target.exists():
            stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
            target = target_root / f"{self.physical_attempt_id}.{stamp}"

        os.replace(str(workspace), str(target))
        fsync_path(target_root)

        # Write a run_attempt_manifest beside the preserved workspace so the
        # cancelled run remains diagnosable (S5 acceptance: "cancel leaves a
        # recoverable workspace and attempt manifest").
        attempt = RunAttemptManifest(
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
            manifest_hash="",  # filled by kernel; saga does not know it here
            params_hash="",
            image_identity={"kind": "unverified_local", "value": "cancelled"},
            mv_contract_version="1",
            final_state="CANCELLED",
            failure_reason="user cancellation",
            produced_at=produced_at_now(self.boot),
            produced_by=ProducedBy(mvd_version=self.boot.mvd_version).to_dict(),
            recovery_hint=(
                "Workspace preserved for inspection. Re-submit the original "
                "run manifest to retry."
            ),
        )
        write_run_attempt_manifest(target, attempt)
        return target

    def _maybe_fault(self, step: CancelStep) -> None:
        if self.after_step_hook is not None:
            self.after_step_hook(step)
