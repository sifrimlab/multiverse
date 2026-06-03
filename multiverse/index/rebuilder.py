"""Three-source rebuild of the SQLite index (STRATEGY R3 / S17).

Sources, in precedence order for any given fact:

* **Journal** — authoritative for intent. The order of records is the
  order of decisions the kernel made.
* **Container engine** — authoritative for runtime existence right *now*.
  Asked by label (``multiverse.run_id``) per-attempt.
* **Artifact tree** — authoritative for outcome. A directory carrying a
  valid ``artifact_manifest.json`` (sidecar-verified) is ``ARTIFACT_SUCCESS``
  regardless of what the journal's last STATE_TRANSITION said about it.

The rebuilder never deletes anything from the filesystem. Per S4 / R5 any
ambiguity is resolved by classifying the run as ``RECOVERY_PENDING`` so the
user can decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..artifact import (ARTIFACT_MANIFEST_FILENAME, ChecksumMismatchError,
                        ManifestCorruptError, ManifestMissingError,
                        read_manifest)
from ..docker_supervisor.client import ContainerEngine, ContainerState
from ..docker_supervisor.labels import LABEL_RUN_ID, label_query
from ..journal import JournalKind, JournalLayout, JournalReader
from ..mvd.state import PrimaryState
from ..promotion.layout import StoreLayout
from .sqlite_index import SqliteIndex


class RebuildOutcome(str, Enum):
    """Per-run classification produced by the rebuild."""

    PROMOTED = "PROMOTED"
    RUNNING_REATTACHED = "RUNNING_REATTACHED"
    RUNNING_DISAPPEARED = "RUNNING_DISAPPEARED"
    PROMOTE_INCOMPLETE = "PROMOTE_INCOMPLETE"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"
    CORRUPT_MANIFEST = "CORRUPT_MANIFEST"
    PENDING_OR_ADMITTED = "PENDING_OR_ADMITTED"


@dataclass
class RebuildClassification:
    physical_attempt_id: str
    outcome: RebuildOutcome
    primary_state: PrimaryState
    artifact_dir: Optional[str] = None
    failure_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RebuildResult:
    classifications: List[RebuildClassification] = field(default_factory=list)
    total_runs: int = 0
    artifact_success: int = 0
    recovery_pending: int = 0
    failed: int = 0
    cancelled: int = 0
    other: int = 0
    truncated_journal_tail: Optional[str] = None
    """Set when the active journal segment ended in a truncated tail."""

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "rebuilt_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
            "total_runs": self.total_runs,
            "artifact_success": self.artifact_success,
            "recovery_pending": self.recovery_pending,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "other": self.other,
            "notes": [
                f"{c.physical_attempt_id}: {c.outcome.value} -> {c.primary_state.value}"
                for c in self.classifications
            ],
        }


def rebuild_index(
    *,
    index: SqliteIndex,
    state_root: Path,
    store: StoreLayout,
    engine: Optional[ContainerEngine] = None,
    truncate: bool = True,
) -> RebuildResult:
    """Replay journal + cross-reference engine + verify artifacts.

    The caller (typically ``multiverse rebuild-index``) is responsible for
    holding the kernel's paused-maintenance lock per R1: this function
    truncates the index before re-populating it, and concurrent writes
    would race.
    """
    result = RebuildResult()
    layout = JournalLayout.at(state_root / "journal")

    if truncate:
        index.truncate_runs()

    # --- Phase 1: replay journal into per-attempt facts. -----------------
    reader = JournalReader(layout)
    replay = reader.replay()
    if replay.truncated_tail_at is not None:
        result.truncated_journal_tail = str(replay.truncated_tail_at)

    facts: Dict[str, _AttemptFacts] = {}
    for record in replay.records:
        attempt = record.physical_attempt_id
        if not attempt:
            continue
        facts.setdefault(attempt, _AttemptFacts(physical_attempt_id=attempt))
        facts[attempt].consume(record)

    # --- Phase 2: classify each attempt. ---------------------------------
    for attempt, attempt_facts in facts.items():
        classification = _classify(attempt_facts, store=store, engine=engine)
        result.classifications.append(classification)
        _tally(result, classification.primary_state)
        index.upsert_run(
            {
                "physical_attempt_id": attempt,
                "logical_run_id": attempt_facts.logical_run_id,
                "primary_state": classification.primary_state.value,
                "failure_reason": classification.failure_reason,
                "artifact_dir": classification.artifact_dir,
                "workspace_dir": attempt_facts.workspace_dir,
                "manifest_path": attempt_facts.manifest_path,
                "cancel_requested": attempt_facts.cancel_requested,
                "submitted_wall_iso": attempt_facts.submitted_wall_iso,
                "last_seq": attempt_facts.last_seq,
                "options": attempt_facts.options,
                "user_id": attempt_facts.user_id,
            }
        )
        for plugin, status in attempt_facts.projections.items():
            index.set_projection(
                physical_attempt_id=attempt,
                plugin=plugin,
                status=status,
            )
        for ev in attempt_facts.reservation_events:
            index.upsert_reservation_event(
                physical_attempt_id=attempt,
                seq=ev["seq"],
                kind=ev["kind"],
                wall_iso=ev["wall_iso"],
                ram_bytes=ev.get("ram_bytes"),
                gpu_index=ev.get("gpu_index"),
                release_reason=ev.get("release_reason"),
            )

    result.total_runs = len(result.classifications)
    index.record_rebuild_report(result.summary_dict())
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _AttemptFacts:
    physical_attempt_id: str
    logical_run_id: Optional[str] = None
    workspace_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    submitted_wall_iso: Optional[str] = None
    last_journal_state: Optional[str] = None
    artifact_dir: Optional[str] = None
    manifest_committed: bool = False
    promote_prepared: bool = False
    quarantine_path: Optional[str] = None
    container_id: Optional[str] = None
    cancel_requested: bool = False
    last_seq: int = 0
    options: Dict[str, Any] = field(default_factory=dict)
    projections: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    reservation_events: List[Dict[str, Any]] = field(default_factory=list)

    def consume(self, record) -> None:
        self.last_seq = max(self.last_seq, record.seq)
        if record.logical_run_id and not self.logical_run_id:
            self.logical_run_id = record.logical_run_id
        # Capture user_id from any record that carries it (G2).
        if record.user_id and not self.user_id:
            self.user_id = record.user_id

        if record.kind is JournalKind.RESERVATION_GRANTED:
            self.reservation_events.append(
                {
                    "seq": record.seq,
                    "kind": "granted",
                    "wall_iso": record.wall_iso,
                    "ram_bytes": record.payload.get("ram_bytes"),
                    "gpu_index": record.payload.get("gpu_index"),
                    "release_reason": None,
                }
            )
        elif record.kind is JournalKind.RESERVATION_RELEASED:
            self.reservation_events.append(
                {
                    "seq": record.seq,
                    "kind": "released",
                    "wall_iso": record.wall_iso,
                    "ram_bytes": None,
                    "gpu_index": None,
                    "release_reason": record.payload.get("reason"),
                }
            )
        elif record.kind is JournalKind.JOB_INTENT:
            self.manifest_path = record.payload.get("manifest_path")
            self.options = dict(record.payload.get("options") or {})
            self.submitted_wall_iso = record.wall_iso
            self.last_journal_state = "PENDING"
        elif record.kind is JournalKind.STATE_TRANSITION:
            self.last_journal_state = record.payload.get("to_state")
        elif record.kind is JournalKind.PROMOTE_PREPARE:
            self.promote_prepared = True
            self.workspace_dir = record.payload.get("workspace_dir")
            self.artifact_dir = record.payload.get("final_artifact_dir")
        elif record.kind is JournalKind.PROMOTE_COMMIT_MANIFEST:
            self.manifest_committed = True
            self.artifact_dir = record.payload.get("artifact_dir", self.artifact_dir)
        elif record.kind is JournalKind.PROMOTION_QUARANTINE:
            self.quarantine_path = record.payload.get("source")
        elif record.kind is JournalKind.CONTAINER_LAUNCH:
            labels = record.payload.get("labels") or {}
            self.workspace_dir = labels.get("multiverse.workspace", self.workspace_dir)
        elif record.kind is JournalKind.CANCEL_REQUESTED:
            self.cancel_requested = True
        elif record.kind is JournalKind.CANCELLED:
            self.last_journal_state = PrimaryState.CANCELLED.value
        elif record.kind is JournalKind.PROJECTION_STATUS:
            plugin = record.payload.get("plugin")
            status = record.payload.get("status")
            if plugin and status:
                self.projections[plugin] = status


def _classify(
    facts: _AttemptFacts,
    *,
    store: StoreLayout,
    engine: Optional[ContainerEngine],
) -> RebuildClassification:
    # Outcome priority: a sidecar-verified artifact manifest wins.
    if facts.artifact_dir:
        manifest_state = _check_artifact_manifest(Path(facts.artifact_dir))
        if manifest_state == "verified":
            return RebuildClassification(
                physical_attempt_id=facts.physical_attempt_id,
                outcome=RebuildOutcome.PROMOTED,
                primary_state=PrimaryState.ARTIFACT_SUCCESS,
                artifact_dir=facts.artifact_dir,
                notes=["sidecar-verified manifest on disk"],
            )
        if manifest_state == "corrupt":
            return RebuildClassification(
                physical_attempt_id=facts.physical_attempt_id,
                outcome=RebuildOutcome.CORRUPT_MANIFEST,
                primary_state=PrimaryState.RECOVERY_PENDING,
                artifact_dir=facts.artifact_dir,
                failure_reason="manifest body did not verify against sidecar",
                notes=["artifact_manifest.json present but checksum mismatch"],
            )

    # Promotion saga half-finished.
    if facts.promote_prepared and not facts.manifest_committed:
        return RebuildClassification(
            physical_attempt_id=facts.physical_attempt_id,
            outcome=RebuildOutcome.PROMOTE_INCOMPLETE,
            primary_state=PrimaryState.RECOVERY_PENDING,
            artifact_dir=facts.artifact_dir,
            failure_reason="promotion prepared but never committed manifest",
            notes=["recovery: PROMOTE_PREPARE without PROMOTE_COMMIT_MANIFEST"],
        )

    # Terminal states declared by the journal itself.
    if facts.last_journal_state == PrimaryState.CANCELLED.value:
        return RebuildClassification(
            physical_attempt_id=facts.physical_attempt_id,
            outcome=RebuildOutcome.ALREADY_TERMINAL,
            primary_state=PrimaryState.CANCELLED,
        )
    if facts.last_journal_state == PrimaryState.FAILED.value:
        return RebuildClassification(
            physical_attempt_id=facts.physical_attempt_id,
            outcome=RebuildOutcome.ALREADY_TERMINAL,
            primary_state=PrimaryState.FAILED,
        )
    if facts.last_journal_state == PrimaryState.RECOVERY_PENDING.value:
        return RebuildClassification(
            physical_attempt_id=facts.physical_attempt_id,
            outcome=RebuildOutcome.ALREADY_TERMINAL,
            primary_state=PrimaryState.RECOVERY_PENDING,
        )

    # Active runtime check via labels.
    if engine is not None and facts.last_journal_state in {
        PrimaryState.RUNNING.value,
        PrimaryState.ADMITTED.value,
        PrimaryState.TRAINING_SUCCEEDED.value,
        PrimaryState.EVALUATING.value,
        PrimaryState.PROMOTING.value,
        PrimaryState.CANCEL_REQUESTED.value,
    }:
        containers = engine.list_by_labels(
            labels=label_query(facts.physical_attempt_id)
        )
        running = [c for c in containers if c.state is ContainerState.RUNNING]
        if running:
            return RebuildClassification(
                physical_attempt_id=facts.physical_attempt_id,
                outcome=RebuildOutcome.RUNNING_REATTACHED,
                primary_state=PrimaryState(facts.last_journal_state),
                notes=[f"reattached container {running[0].container_id}"],
            )
        return RebuildClassification(
            physical_attempt_id=facts.physical_attempt_id,
            outcome=RebuildOutcome.RUNNING_DISAPPEARED,
            primary_state=PrimaryState.RECOVERY_PENDING,
            failure_reason="journal RUNNING-class state but no live container",
            notes=["recovery: container disappeared between boots"],
        )

    # PENDING / ADMITTED with no journal terminal record — fall through.
    if facts.last_journal_state in {
        PrimaryState.PENDING.value,
        PrimaryState.ADMITTED.value,
    }:
        return RebuildClassification(
            physical_attempt_id=facts.physical_attempt_id,
            outcome=RebuildOutcome.PENDING_OR_ADMITTED,
            primary_state=PrimaryState.RECOVERY_PENDING,
            failure_reason="journal stopped at PENDING/ADMITTED; assume crashed pre-run",
        )

    # Fallback: surface whatever the journal said last, defaulting to
    # RECOVERY_PENDING so the user sees the run rather than losing it.
    try:
        state = PrimaryState(facts.last_journal_state or "RECOVERY_PENDING")
    except ValueError:
        state = PrimaryState.RECOVERY_PENDING
    return RebuildClassification(
        physical_attempt_id=facts.physical_attempt_id,
        outcome=RebuildOutcome.ALREADY_TERMINAL,
        primary_state=state,
    )


def _check_artifact_manifest(artifact_dir: Path) -> str:
    if not (artifact_dir / ARTIFACT_MANIFEST_FILENAME).is_file():
        return "missing"
    try:
        read_manifest(artifact_dir)
        return "verified"
    except ChecksumMismatchError:
        return "corrupt"
    except (ManifestMissingError, ManifestCorruptError):
        return "corrupt"


def _tally(result: RebuildResult, state: PrimaryState) -> None:
    if state is PrimaryState.ARTIFACT_SUCCESS:
        result.artifact_success += 1
    elif state is PrimaryState.RECOVERY_PENDING:
        result.recovery_pending += 1
    elif state is PrimaryState.FAILED:
        result.failed += 1
    elif state is PrimaryState.CANCELLED:
        result.cancelled += 1
    else:
        result.other += 1
