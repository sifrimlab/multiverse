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
    """Did the probe itself succeed? ``SKIPPED`` means the service was not
    configured, not that the probe failed."""

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class CleanupResult(str, Enum):
    """Did the probe manage to remove the scratch it created this run?"""

    CLEAN = "clean"
    LEAKED = "leaked"
    CLEANUP_FAILED = "cleanup_failed"


class LeakInventoryResult(str, Enum):
    """Did the probe find *prior* leaked entries left in its namespace?"""

    NONE = "none"
    LEAKS = "leaks"
    INVENTORY_FAILED = "inventory_failed"


@dataclass
class ProbeReport:
    """One health probe's three-column result plus an optional detail line.

    Attributes:
        name: Probe identifier surfaced in the doctor report.
        probe: Whether the probe ran and passed.
        cleanup: Whether the probe's own scratch was reclaimed.
        leak: Whether stale entries from earlier runs were found.
        leak_count: Number of leaked entries counted by the inventory.
        detail: Human-readable summary or error string.
    """

    name: str
    probe: ProbeOutcome
    cleanup: CleanupResult
    leak: LeakInventoryResult
    leak_count: int = 0
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Render as the JSON-serialisable row used by ``doctor --json``."""
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
    """Round-trip a workspace under the reserved health-probe namespace.

    Creates a probe workspace, writes a marker, then removes only its own
    entry. Older sibling entries left under the reserved name are counted
    as leaks (a previous probe or sweeper failed to clean them).

    Args:
        workspaces_root: ``store/workspaces/`` root; the reserved
            ``__mvd_health_probe__`` namespace is created beneath it.

    Returns:
        A :class:`ProbeReport` named ``workspace_dir`` carrying the
        probe/cleanup/leak triple.
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
    """Remove expired entries from the reserved health-probe namespaces.

    Only operates inside the reserved namespaces. Per R12 Tier-1 GC this
    is one of the closed-list paths the daemon is allowed to clean.

    Args:
        workspaces_root: ``store/workspaces/`` root holding the reserved
            ``__mvd_health_probe__`` namespace.
        ttl_seconds: Entries older than this (by mtime) are removed.
        now: Override for the current epoch time; injected by tests.

    Returns:
        Per-namespace count of removed entries (currently only
        ``workspace_dir``).
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
    """Count entries older than the TTL directly under ``probe_root``."""
    if not probe_root.is_dir():
        return 0
    now = time.time()
    count = 0
    for entry in probe_root.iterdir():
        age = now - entry.stat().st_mtime
        if age > HEALTH_PROBE_TTL_SECONDS:
            count += 1
    return count
