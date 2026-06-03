"""MLflow target protocol — abstraction for tests and production."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Dict, List, Mapping, Optional, Protocol,
                    runtime_checkable)


class SyncOutcome(str, Enum):
    SYNCED = "TRACKING_SYNCED"
    SYNC_FAILED = "TRACKING_SYNC_FAILED"
    NOT_CONFIGURED = "TRACKING_NOT_CONFIGURED"
    NOT_APPLICABLE = "TRACKING_NOT_APPLICABLE"


@dataclass
class SyncResult:
    physical_attempt_id: str
    outcome: SyncOutcome
    target_run_id: Optional[str] = None
    failure_reason: Optional[str] = None
    metrics_logged: int = 0
    artifacts_logged: int = 0


@runtime_checkable
class MLflowTarget(Protocol):
    """Minimum surface the sync plugin needs from MLflow.

    Tests construct an in-memory implementation; production wraps the
    real ``mlflow`` SDK in a tiny adapter.
    """

    name: str

    def create_run(
        self,
        *,
        experiment_name: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> str:
        """Create or upsert a run; return its target-side run id."""
        ...

    def log_params(self, *, run_id: str, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, *, run_id: str, metrics: Mapping[str, float]) -> None: ...

    def log_artifact(self, *, run_id: str, path: str) -> None: ...

    def set_terminal_status(
        self,
        *,
        run_id: str,
        status: str,
    ) -> None:
        """``status`` is one of FINISHED, FAILED, KILLED — MLflow's
        canonical run-end strings."""
        ...


# ---------------------------------------------------------------------------
# In-memory fake for tests
# ---------------------------------------------------------------------------


@dataclass
class _InMemoryRun:
    target_run_id: str
    experiment: str
    run_name: str
    tags: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    terminal_status: Optional[str] = None


@dataclass
class InMemoryMLflowTarget:
    """Deterministic fake. Use ``fail_on_create=True`` to simulate an
    MLflow outage at the create-run step."""

    name: str = "in-memory-mlflow"
    runs: Dict[str, _InMemoryRun] = field(default_factory=dict)
    fail_on_create: bool = False
    fail_on_log: bool = False
    _next_id: int = 0

    def _alloc(self) -> str:
        self._next_id += 1
        return f"run-{self._next_id:06d}"

    def create_run(self, *, experiment_name, run_name, tags) -> str:
        if self.fail_on_create:
            raise ConnectionError("MLflow tracking server unreachable")
        rid = self._alloc()
        self.runs[rid] = _InMemoryRun(
            target_run_id=rid,
            experiment=experiment_name,
            run_name=run_name,
            tags=dict(tags),
        )
        return rid

    def log_params(self, *, run_id, params):
        if self.fail_on_log:
            raise ConnectionError("MLflow log_params failed")
        self.runs[run_id].params.update(params)

    def log_metrics(self, *, run_id, metrics):
        if self.fail_on_log:
            raise ConnectionError("MLflow log_metrics failed")
        self.runs[run_id].metrics.update(metrics)

    def log_artifact(self, *, run_id, path):
        if self.fail_on_log:
            raise ConnectionError("MLflow log_artifact failed")
        self.runs[run_id].artifacts.append(path)

    def set_terminal_status(self, *, run_id, status):
        self.runs[run_id].terminal_status = status
