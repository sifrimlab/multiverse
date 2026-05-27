"""Build a GC plan from candidates + retention policy.

The plan is a per-candidate decision: ``would_delete`` / ``would_keep``
plus the reason. ``apply_plan`` then either prints the dry-run report or,
under ``--apply``, performs the moves.

Retention policy defaults to "infinite" — no retention — until the user
opts in. This implements R12's "a user can run for a year without any
artifact ever being auto-deleted".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from .candidates import CandidateKind, GcCandidate


class PlanReason(str, Enum):
    WOULD_DELETE = "would_delete"
    KEEP_NO_RETENTION = "keep: retention policy infinite for this kind"
    KEEP_NOT_AGED = "keep: not aged past retention threshold"
    KEEP_NO_OWNER_TOKEN = "keep: no .mvd_owner; refusing to delete"
    KEEP_NO_EXPORT = "keep: no export bundle and --no-export-required not set"
    KEEP_PROMOTED_PROTECTED = "keep: promoted artifact protected by default policy"


@dataclass
class RetentionPolicy:
    """Per-kind retention windows. ``None`` means infinite (never auto-
    aged into deletion)."""

    failed_workspaces_seconds: Optional[int] = None
    cancelled_workspaces_seconds: Optional[int] = None
    quarantine_seconds: Optional[int] = None
    # Promoted artifacts are NEVER auto-aged. The strategy is explicit:
    # default GC cannot delete promoted user artifacts (R12 acceptance).
    promoted_artifacts_seconds: None = field(default=None, init=False)

    def threshold_for(self, kind: CandidateKind) -> Optional[int]:
        if kind is CandidateKind.PROMOTED_ARTIFACT:
            return None  # never
        if kind is CandidateKind.FAILED_WORKSPACE:
            return self.failed_workspaces_seconds
        if kind is CandidateKind.CANCELLED_WORKSPACE:
            return self.cancelled_workspaces_seconds
        if kind is CandidateKind.QUARANTINE:
            return self.quarantine_seconds
        return None


@dataclass
class PlanEntry:
    candidate: GcCandidate
    reason: PlanReason
    note: Optional[str] = None

    @property
    def would_delete(self) -> bool:
        return self.reason is PlanReason.WOULD_DELETE


@dataclass
class GcPlan:
    entries: List[PlanEntry] = field(default_factory=list)

    @property
    def to_delete(self) -> List[PlanEntry]:
        return [e for e in self.entries if e.would_delete]

    @property
    def to_keep(self) -> List[PlanEntry]:
        return [e for e in self.entries if not e.would_delete]

    def by_kind(self) -> Dict[CandidateKind, List[PlanEntry]]:
        out: Dict[CandidateKind, List[PlanEntry]] = {k: [] for k in CandidateKind}
        for e in self.entries:
            out[e.candidate.kind].append(e)
        return out


def build_plan(
    candidates: Sequence[GcCandidate],
    *,
    policy: RetentionPolicy,
    require_export: bool = True,
    apply_to_promoted: bool = False,
) -> GcPlan:
    """Compute deletion decisions for every candidate.

    ``apply_to_promoted=False`` (the default) hard-protects every
    promoted artifact regardless of retention. To delete a promoted
    artifact a user must explicitly pass ``--apply-to-promoted``, AND it
    must have an export, AND its owner token must be present. Even with
    all three, the strategy still requires manual ``--apply``.
    """
    plan = GcPlan()
    for candidate in candidates:
        plan.entries.append(_decide(candidate, policy, require_export, apply_to_promoted))
    return plan


def _decide(
    candidate: GcCandidate,
    policy: RetentionPolicy,
    require_export: bool,
    apply_to_promoted: bool,
) -> PlanEntry:
    # Promoted artifacts are protected by default.
    if candidate.kind is CandidateKind.PROMOTED_ARTIFACT and not apply_to_promoted:
        return PlanEntry(candidate, PlanReason.KEEP_PROMOTED_PROTECTED)

    threshold = policy.threshold_for(candidate.kind)
    if threshold is None:
        return PlanEntry(
            candidate,
            PlanReason.KEEP_NO_RETENTION,
            note=f"no retention threshold configured for {candidate.kind.value}",
        )
    if candidate.age_seconds < threshold:
        return PlanEntry(
            candidate,
            PlanReason.KEEP_NOT_AGED,
            note=f"age {int(candidate.age_seconds)}s < threshold {threshold}s",
        )
    if candidate.owner_token is None:
        return PlanEntry(candidate, PlanReason.KEEP_NO_OWNER_TOKEN)
    if require_export and not candidate.has_export:
        return PlanEntry(
            candidate,
            PlanReason.KEEP_NO_EXPORT,
            note="missing EXPORTED marker; bundle this run before gc",
        )
    return PlanEntry(candidate, PlanReason.WOULD_DELETE)
