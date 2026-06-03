"""In-memory run registry (STRATEGY R3 / R6).

Per Milestone 7 the registry is rebuilt from the journal on boot — SQLite
indexing is Milestone 8. The registry is the kernel's working set: every
state mutation goes through it, every API query reads from it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .state import PROJECTION_STATUSES, PrimaryState


@dataclass
class RunRecord:
    """One row in the in-memory run registry."""

    physical_attempt_id: str
    logical_run_id: Optional[str]
    manifest_path: Optional[str]
    submitted_at_monotonic_ns: int
    submitted_wall_iso: str
    primary_state: PrimaryState = PrimaryState.PENDING
    cancel_requested: bool = False
    failure_reason: Optional[str] = None
    artifact_dir: Optional[str] = None
    workspace_dir: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    projections: Dict[str, str] = field(
        default_factory=lambda: {
            "mlflow": "TRACKING_NOT_CONFIGURED",
            "optuna": "TRACKING_NOT_APPLICABLE",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "physical_attempt_id": self.physical_attempt_id,
            "logical_run_id": self.logical_run_id,
            "primary_state": self.primary_state.value,
            "cancel_requested": self.cancel_requested,
            "failure_reason": self.failure_reason,
            "artifact_dir": self.artifact_dir,
            "workspace_dir": self.workspace_dir,
            "manifest_path": self.manifest_path,
            "submitted_wall_iso": self.submitted_wall_iso,
            "projections": dict(self.projections),
            "options": dict(self.options),
        }


@dataclass
class RunRegistry:
    """Map of physical_attempt_id → RunRecord with simple listing helpers."""

    records: Dict[str, RunRecord] = field(default_factory=dict)
    _listeners: List[Callable[[RunRecord], None]] = field(default_factory=list)

    def add(self, record: RunRecord) -> None:
        """Register a run and notify listeners."""
        self.records[record.physical_attempt_id] = record
        self._fire(record)

    def get(self, physical_attempt_id: str) -> RunRecord:
        """Return the run record; raises ``KeyError`` if unknown."""
        try:
            return self.records[physical_attempt_id]
        except KeyError as exc:
            raise KeyError(f"no run with id {physical_attempt_id!r}") from exc

    def has(self, physical_attempt_id: str) -> bool:
        """Return whether ``physical_attempt_id`` is in the registry."""
        return physical_attempt_id in self.records

    def list(
        self,
        *,
        state: Optional[PrimaryState] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[RunRecord]:
        """List runs, optionally filtered by state or logical run id."""
        out = list(self.records.values())
        if state is not None:
            out = [r for r in out if r.primary_state is state]
        if logical_run_id is not None:
            out = [r for r in out if r.logical_run_id == logical_run_id]
        out.sort(key=lambda r: r.submitted_at_monotonic_ns)
        return out

    def add_listener(self, listener: Callable[[RunRecord], None]) -> None:
        """Register a callback invoked on ``add``; listener errors are swallowed."""
        self._listeners.append(listener)

    def _fire(self, record: RunRecord) -> None:
        for listener in self._listeners:
            try:
                listener(record)
            except Exception:  # pragma: no cover — listener bugs must not
                # propagate into the kernel.
                continue


def new_run_record(
    *,
    physical_attempt_id: str,
    manifest_path: Optional[str],
    logical_run_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> RunRecord:
    """Create a new run record with submit-time timestamps."""
    return RunRecord(
        physical_attempt_id=physical_attempt_id,
        logical_run_id=logical_run_id,
        manifest_path=manifest_path,
        submitted_at_monotonic_ns=time.monotonic_ns(),
        submitted_wall_iso=datetime.now(timezone.utc).astimezone().isoformat(),
        options=dict(options or {}),
    )


def assert_projection_status_valid(plugin: str, status: str) -> None:
    allowed = PROJECTION_STATUSES.get(plugin)
    if allowed is None:
        raise ValueError(f"unknown projection plugin {plugin!r}")
    if status not in allowed:
        raise ValueError(
            f"projection status {status!r} not allowed for plugin {plugin!r}; "
            f"allowed: {sorted(allowed)}"
        )
