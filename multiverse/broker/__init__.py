"""Resource broker — admission control and pressure observation (STRATEGY S9 / R11 / Milestone 13).

The resource broker enforces admission control (leases/reservations) before
each physical attempt is launched. It observes host state at three rates:

* **At admission** — re-poll RAM/VRAM/disk/inodes/FDs; refuse the job if
  the live total minus the reservation ledger is below the job's request.
* **Continuous** — 5 s tick during RUNNING; update a rolling pressure
  score per resource.
* **On Docker events** — OOM kill, paused, exited.

The broker NEVER preempts running containers in local mode (R11). It only
*stops admitting* new jobs under pressure. The user retains authority over
running work.

The reservation ledger is reconstructed from the append-only journal on
startup (``reconstruct_ledger_from_journal``), so a crash never leaves stale
leases in memory.
"""

from .broker import (AdmissionDecision, AdmissionOutcome,
                     ContinuousObservation, OomEvent, PressureEvent,
                     ReservationLedger, ResourceBroker,
                     reconstruct_ledger_from_journal)
from .observer import (HostMetrics, HostObserver, InMemoryHostObserver,
                       ResourceRequest)
from .pressure import PressureMode, PressureThresholds

__all__ = [
    "AdmissionDecision",
    "AdmissionOutcome",
    "ContinuousObservation",
    "HostMetrics",
    "HostObserver",
    "InMemoryHostObserver",
    "OomEvent",
    "PressureEvent",
    "PressureMode",
    "PressureThresholds",
    "ReservationLedger",
    "ResourceBroker",
    "ResourceRequest",
    "reconstruct_ledger_from_journal",
]
