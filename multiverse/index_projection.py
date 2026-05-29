"""Read-mostly projection surface (STRATEGY M5).

The kernel's authoritative records are the journal and the artifact
tree; SQLite is a *projection* — a fast queryable cache that is fully
rebuildable from those two surfaces. This module is the public
read-mostly facade over :mod:`multiverse.index`. Callers use it when
they want to ask "what does the projection say about run X?" without
opening the writer surface.

The write surface still exists for one reason: the kernel's index-
projection plugin advances the projection seq-by-seq as the kernel
appends to the journal. That plugin uses :class:`multiverse.index.
SqliteIndex` directly. Every other caller — the GUI, the CLI, doctor —
should depend on this module.

Why a separate file from :mod:`multiverse.index`? The read surface
needs to be obvious. ``from multiverse.index_projection import
get_run`` makes the read-mostly contract part of the import line.

Out of scope here:
* Datasets and models metadata. Those live in
  :mod:`multiverse.registry_db` because they are user-supplied
  registrations, not derivations of the journal. The naming makes the
  distinction explicit: this module is the *projection* of in-kernel
  state; :mod:`registry_db` is the *registry* of user-managed assets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .index import INDEX_FILENAME, SqliteIndex, open_index
from .journal import JournalKind, JournalLayout, JournalReader
from .mvd.state import PrimaryState


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def get_run(
    state_root: Path, *, physical_attempt_id: str
) -> Optional[Dict[str, Any]]:
    """Return the projection's view of one attempt, or ``None`` if the
    projection has not seen it yet.

    Read-only. Opens the SQLite file in WAL mode and closes it on
    return; safe to call concurrently with the kernel's writer.
    """
    db_path = Path(state_root) / INDEX_FILENAME
    if not db_path.is_file():
        return None
    with open_index(db_path) as idx:
        return idx.get_run(physical_attempt_id)


def list_runs(
    state_root: Path,
    *,
    primary_state: Optional[str] = None,
    logical_run_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read-only list-runs over the projection."""
    db_path = Path(state_root) / INDEX_FILENAME
    if not db_path.is_file():
        return []
    with open_index(db_path) as idx:
        return idx.list_runs(
            primary_state=primary_state,
            logical_run_id=logical_run_id,
            limit=limit,
        )


def projections_for(
    state_root: Path, *, physical_attempt_id: str
) -> Dict[str, str]:
    db_path = Path(state_root) / INDEX_FILENAME
    if not db_path.is_file():
        return {}
    with open_index(db_path) as idx:
        return idx.projections_for(physical_attempt_id)


def reservation_events_for(
    state_root: Path, *, physical_attempt_id: str
) -> List[Dict[str, Any]]:
    """Return the reservation timeline for one attempt from the projection.

    Each entry has: ``seq``, ``kind`` (``"granted"``/``"released"``),
    ``wall_iso``, ``ram_bytes``, ``gpu_index``, ``release_reason``.
    Returns ``[]`` if the projection does not exist or has no events for
    this attempt.
    """
    db_path = Path(state_root) / INDEX_FILENAME
    if not db_path.is_file():
        return []
    with open_index(db_path) as idx:
        return idx.list_reservation_events(physical_attempt_id)


# ---------------------------------------------------------------------------
# Consistency verification (read-only)
# ---------------------------------------------------------------------------


@dataclass
class ProjectionDrift:
    """One disagreement between journal + projection."""

    physical_attempt_id: str
    kind: str  # "missing_in_projection" | "stale_state" | "orphan_in_projection"
    journal_state: Optional[str] = None
    projection_state: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "physical_attempt_id": self.physical_attempt_id,
            "kind": self.kind,
            "journal_state": self.journal_state,
            "projection_state": self.projection_state,
            "detail": self.detail,
        }


@dataclass
class ProjectionVerifyReport:
    """Aggregate of one read-only consistency check."""

    drifts: List[ProjectionDrift] = field(default_factory=list)
    runs_in_journal: int = 0
    runs_in_projection: int = 0
    truncated_journal_tail: Optional[str] = None
    generated_iso: str = ""

    @property
    def in_sync(self) -> bool:
        return not self.drifts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_iso": self.generated_iso,
            "runs_in_journal": int(self.runs_in_journal),
            "runs_in_projection": int(self.runs_in_projection),
            "drift_count": len(self.drifts),
            "truncated_journal_tail": self.truncated_journal_tail,
            "drifts": [d.to_dict() for d in self.drifts],
        }


def verify_projection_against_journal(
    state_root: Path,
) -> ProjectionVerifyReport:
    """Walk the journal, walk the projection, report disagreements.

    Definitions:
        * **missing_in_projection**: the journal has a JOB_INTENT for
          this attempt, but the projection has no row.
        * **stale_state**: both sides know the attempt; the projection's
          ``primary_state`` does not match the journal's last
          STATE_TRANSITION (or CANCELLED record).
        * **orphan_in_projection**: the projection has a row that does
          not appear anywhere in the journal.

    The check is read-only; tests sometimes call it with no journal at
    all, in which case the report is empty.
    """
    report = ProjectionVerifyReport(
        generated_iso=datetime.now(timezone.utc).astimezone().isoformat()
    )
    journal_root = Path(state_root) / "journal"
    db_path = Path(state_root) / INDEX_FILENAME

    journal_states: Dict[str, str] = {}
    journal_seen: Dict[str, bool] = {}
    if journal_root.is_dir():
        replay = JournalReader(JournalLayout.at(journal_root)).replay()
        if replay.truncated_tail_at is not None:
            report.truncated_journal_tail = str(replay.truncated_tail_at)
        for record in replay.records:
            attempt = record.physical_attempt_id
            if not attempt:
                continue
            if record.kind is JournalKind.JOB_INTENT:
                journal_seen[attempt] = True
                journal_states.setdefault(attempt, PrimaryState.PENDING.value)
            elif record.kind is JournalKind.STATE_TRANSITION:
                to_state = record.payload.get("to_state")
                if to_state:
                    journal_states[attempt] = str(to_state)
            elif record.kind is JournalKind.CANCELLED:
                journal_states[attempt] = PrimaryState.CANCELLED.value

    report.runs_in_journal = len(journal_seen)

    projection_rows: Dict[str, str] = {}
    if db_path.is_file():
        with open_index(db_path) as idx:
            for row in idx.list_runs():
                projection_rows[row["physical_attempt_id"]] = str(
                    row.get("primary_state") or ""
                )
    report.runs_in_projection = len(projection_rows)

    for attempt, journal_state in journal_states.items():
        if attempt not in projection_rows:
            report.drifts.append(
                ProjectionDrift(
                    physical_attempt_id=attempt,
                    kind="missing_in_projection",
                    journal_state=journal_state,
                    detail="JOB_INTENT in journal, no row in projection",
                )
            )
            continue
        projection_state = projection_rows[attempt]
        if projection_state != journal_state:
            report.drifts.append(
                ProjectionDrift(
                    physical_attempt_id=attempt,
                    kind="stale_state",
                    journal_state=journal_state,
                    projection_state=projection_state,
                    detail=(
                        "projection's primary_state does not match the "
                        "journal's last terminal/transition record"
                    ),
                )
            )

    for attempt, projection_state in projection_rows.items():
        if attempt not in journal_states:
            report.drifts.append(
                ProjectionDrift(
                    physical_attempt_id=attempt,
                    kind="orphan_in_projection",
                    projection_state=projection_state,
                    detail="row in projection but no JOB_INTENT in journal",
                )
            )

    return report


__all__ = [
    "INDEX_FILENAME",
    "ProjectionDrift",
    "ProjectionVerifyReport",
    "SqliteIndex",
    "get_run",
    "list_runs",
    "open_index",
    "projections_for",
    "verify_projection_against_journal",
]
