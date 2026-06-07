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
    """One row in the in-memory run registry.

    Attributes:
        physical_attempt_id: Unique identity for this execution attempt.
        logical_run_id: Groups retries/resumes of the same recipe; ``None``
            until the executor resolves it from the job options.
        manifest_path: Path to the manifest that produced this run, if known.
        submitted_at_monotonic_ns: ``time.monotonic_ns()`` at submission; used
            for deterministic list ordering within a single kernel process.
        submitted_wall_iso: ISO-8601 wall-clock timestamp at submission,
            recorded in the journal for human-readable provenance.
        primary_state: Current position in the run-state machine; starts at
            PENDING and advances only through :meth:`Kernel.transition`.
        cancel_requested: Set to ``True`` when ``cancel_run`` is called; the
            executor polls this flag between steps to honour the cancellation
            saga.
        failure_reason: Human-readable reason for terminal FAILED,
            PROMOTION_FAILED, or EVALUATION_FAILED states; ``None`` otherwise.
        artifact_dir: Absolute path to the promoted artifact bundle in the
            artifact store; populated after ARTIFACT_SUCCESS.
        workspace_dir: Absolute path to the in-flight workspace directory
            under ``store/workspaces/``; populated at container launch.
        options: Executor-specific job options carried with the run (image,
            dataset path, resource request, etc.).
        projections: Per-plugin sync status (e.g.
            ``{"mlflow": "TRACKING_PENDING"}``); a projection is never the
            source of run truth — the journal is authoritative.
    """

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
        """Serialize the record to the dict shape the API returns to clients."""
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
        """Return the run record for an attempt.

        Args:
            physical_attempt_id: The attempt to look up.

        Returns:
            The matching :class:`RunRecord`.

        Raises:
            KeyError: If no run with that id is registered.
        """
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
        """List runs, optionally filtered, ordered by submit time.

        Args:
            state: If set, keep only runs in this primary state.
            logical_run_id: If set, keep only attempts of this logical run.

        Returns:
            Matching records sorted by monotonic submit time.
        """
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
        """Invoke all registered listeners; swallow individual listener errors."""
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
    """Create a new run record stamped with submit-time timestamps.

    Args:
        physical_attempt_id: Identity of this execution attempt.
        manifest_path: Path to the manifest that produced the run, if known.
        logical_run_id: Logical run grouping retries/resumes of one recipe.
        options: Executor-specific job options carried with the run.

    Returns:
        A :class:`RunRecord` in the PENDING state.
    """
    return RunRecord(
        physical_attempt_id=physical_attempt_id,
        logical_run_id=logical_run_id,
        manifest_path=manifest_path,
        submitted_at_monotonic_ns=time.monotonic_ns(),
        submitted_wall_iso=datetime.now(timezone.utc).astimezone().isoformat(),
        options=dict(options or {}),
    )


def assert_projection_status_valid(plugin: str, status: str) -> None:
    """Validate a projection status name against its plugin's allowed set.

    Args:
        plugin: Projection plugin name (e.g. ``"mlflow"``, ``"optuna"``).
        status: Candidate status name to check.

    Raises:
        ValueError: If the plugin is unknown or the status is not allowed
            for that plugin.
    """
    allowed = PROJECTION_STATUSES.get(plugin)
    if allowed is None:
        raise ValueError(f"unknown projection plugin {plugin!r}")
    if status not in allowed:
        raise ValueError(
            f"projection status {status!r} not allowed for plugin {plugin!r}; "
            f"allowed: {sorted(allowed)}"
        )
