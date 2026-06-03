"""Hidden-namespace health probes + sweeper (STRATEGY R9).

Doctor probes that touch external services (MLflow, Optuna, Docker, the
workspace tree) write only inside reserved namespaces. Each probe reports
three columns:

* ``probe_result`` — pass / fail / skipped (service not configured).
* ``cleanup_result`` — clean / leaked / cleanup_failed.
* ``leak_inventory`` — leaks_<n> / none / inventory_failed.

``mvd-health-sweeper`` walks the reserved namespaces and removes entries
older than the TTL. Doctor's ``--repair-health-probes`` invokes the
sweeper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

HEALTH_PROBE_TTL_SECONDS = 3600

HEALTH_PROBE_NAMESPACES = {
    "mlflow_experiment": "__mvd_health_probe__",
    "optuna_study_prefix": "__mvd_health_probe__",  # plus rfc3339 suffix
    "docker_container_prefix": "multiverse_health_probe_",  # plus rfc3339
    "docker_label": "multiverse.health_probe",
    "workspace_dir": "__mvd_health_probe__",
}


class ProbeOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class CleanupResult(str, Enum):
    CLEAN = "clean"
    LEAKED = "leaked"
    CLEANUP_FAILED = "cleanup_failed"


class LeakInventoryResult(str, Enum):
    NONE = "none"
    LEAKS = "leaks"
    INVENTORY_FAILED = "inventory_failed"


@dataclass
class ProbeReport:
    name: str
    probe: ProbeOutcome
    cleanup: CleanupResult
    leak: LeakInventoryResult
    leak_count: int = 0
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "name": self.name,
            "probe": self.probe.value,
            "cleanup": self.cleanup.value,
            "leak": self.leak.value,
            "leak_count": int(self.leak_count),
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out


# ---------------------------------------------------------------------------
# Workspace health probe (real, exercised in tests)
# ---------------------------------------------------------------------------


def probe_workspace_directory(workspaces_root: Path) -> ProbeReport:
    """Create a workspace under the reserved name, drop a probe file,
    remove the workspace, and report. Counts unrelated stale entries
    under the reserved name as leaks.
    """
    workspaces_root.mkdir(parents=True, exist_ok=True)
    probe_root = workspaces_root / HEALTH_PROBE_NAMESPACES["workspace_dir"]
    probe_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ%f")
    target = probe_root / f"probe-{stamp}"
    cleanup = CleanupResult.CLEAN
    probe_outcome = ProbeOutcome.PASS
    detail: Optional[str] = None
    try:
        target.mkdir(parents=True, exist_ok=False)
        (target / "marker").write_bytes(b"ok")
    except OSError as exc:
        probe_outcome = ProbeOutcome.FAIL
        detail = f"{type(exc).__name__}: {exc}"

    # Cleanup our own entry only.
    if target.exists():
        try:
            for child in target.iterdir():
                child.unlink()
            target.rmdir()
        except OSError as exc:
            cleanup = CleanupResult.CLEANUP_FAILED
            detail = (detail or "") + f" cleanup: {exc}"

    # Leak inventory — count older sibling entries.
    leak_count = _count_expired_under(probe_root)
    leak = LeakInventoryResult.LEAKS if leak_count > 0 else LeakInventoryResult.NONE
    return ProbeReport(
        name="workspace_dir",
        probe=probe_outcome,
        cleanup=cleanup,
        leak=leak,
        leak_count=leak_count,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------


def sweep_expired_health_probes(
    workspaces_root: Path,
    *,
    ttl_seconds: int = HEALTH_PROBE_TTL_SECONDS,
    now: Optional[float] = None,
) -> Dict[str, int]:
    """Walk reserved health-probe namespaces and remove entries older
    than ``ttl_seconds``. Returns a per-namespace count of removed
    entries.

    Only operates inside the reserved namespaces. Per R12 Tier-1 GC this
    is one of the closed-list paths the daemon is allowed to clean.
    """
    now_ts = time.time() if now is None else float(now)
    removed: Dict[str, int] = {}

    probe_root = workspaces_root / HEALTH_PROBE_NAMESPACES["workspace_dir"]
    if not probe_root.is_dir():
        return {"workspace_dir": 0}

    n = 0
    for entry in probe_root.iterdir():
        age = now_ts - entry.stat().st_mtime
        if age <= ttl_seconds:
            continue
        try:
            if entry.is_dir():
                for child in entry.iterdir():
                    child.unlink()
                entry.rmdir()
            else:
                entry.unlink()
            n += 1
        except OSError:
            continue
    removed["workspace_dir"] = n
    return removed


def _count_expired_under(probe_root: Path) -> int:
    if not probe_root.is_dir():
        return 0
    now = time.time()
    count = 0
    for entry in probe_root.iterdir():
        age = now - entry.stat().st_mtime
        if age > HEALTH_PROBE_TTL_SECONDS:
            count += 1
    return count
