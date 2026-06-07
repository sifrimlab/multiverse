"""Manual-first garbage collection (STRATEGY S10 / R12 / Milestone 12).

Two tiers with **hard boundaries**:

* **Tier 1** — kernel-internal scratch. A closed list of paths the kernel
  is allowed to auto-clean (rotated journal segments older than the
  retention window, health-probe namespaces, ``multiverse.health_probe``
  Docker objects). The CI test enumerates this list verbatim.
* **Tier 2** — user-visible artifacts. *Manual only*. Dry-run by default;
  ``--apply`` is required to actually delete. Every deletion is gated by
  three checks: owner-token match, retention age exceeded, and an export
  bundle exists (or the user passed ``--no-export-required``).

The GC plugin is a separate process. The kernel only authorises (by
checking ownership tokens); it does not itself perform user-visible
deletions.
"""

from .apply import GcResult, apply_plan, write_dry_run_report
from .candidates import CandidateKind, GcCandidate, enumerate_candidates
from .errors import GcError, GcGateError, NotOwnedError
from .plan import GcPlan, PlanReason, RetentionPolicy, build_plan
from .tier1 import TIER1_PATHS, Tier1Result, sweep_tier1

__all__ = [
    "CandidateKind",
    "GcCandidate",
    "GcError",
    "GcGateError",
    "GcPlan",
    "GcResult",
    "NotOwnedError",
    "PlanReason",
    "RetentionPolicy",
    "TIER1_PATHS",
    "Tier1Result",
    "apply_plan",
    "build_plan",
    "enumerate_candidates",
    "sweep_tier1",
    "write_dry_run_report",
]
