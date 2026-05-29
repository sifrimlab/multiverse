"""multiverse doctor — diagnostics, storage probes, health-probe namespaces.

STRATEGY S11 / S17 / R8 / R9 / Milestone 11.

The doctor is a *read-only* plugin (R1) by default. ``--repair`` modes are
explicitly enumerated; the default report never mutates user-visible state.
Health probes (R9) write only inside hidden namespaces (``__mvd_health_probe__``)
with TTL cleanup driven by ``mvd-health-sweeper``.
"""

from .projection_probe import probe_projection_consistency
from .reservation_probe import (
    DEFAULT_STALE_AFTER_SECONDS,
    StuckReservation,
    probe_reservation_ledger,
)
from .health_probes import (
    HEALTH_PROBE_NAMESPACES,
    HEALTH_PROBE_TTL_SECONDS,
    CleanupResult,
    LeakInventoryResult,
    ProbeOutcome,
    ProbeReport,
    sweep_expired_health_probes,
)
from .report import (
    DoctorReport,
    DoctorSection,
    SectionStatus,
)
from .storage_probes import (
    BLOCKED,
    CloudSyncMarkerError,
    DANGEROUS,
    DEGRADED,
    StorageLevel,
    StorageProbe,
    StorageProbeResult,
    StorageReport,
    SUPPORTED,
    run_storage_probes,
)

__all__ = [
    "BLOCKED",
    "CleanupResult",
    "CloudSyncMarkerError",
    "DANGEROUS",
    "DEGRADED",
    "DoctorReport",
    "DoctorSection",
    "HEALTH_PROBE_NAMESPACES",
    "HEALTH_PROBE_TTL_SECONDS",
    "LeakInventoryResult",
    "ProbeOutcome",
    "ProbeReport",
    "SectionStatus",
    "StorageLevel",
    "StorageProbe",
    "StorageProbeResult",
    "StorageReport",
    "SUPPORTED",
    "DEFAULT_STALE_AFTER_SECONDS",
    "StuckReservation",
    "probe_projection_consistency",
    "probe_reservation_ledger",
    "run_storage_probes",
    "sweep_expired_health_probes",
]
