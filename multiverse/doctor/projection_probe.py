"""Doctor probe: projection ↔ journal consistency (STRATEGY M5).

The kernel writes to the journal first and advances the SQLite
projection second. A divergence between the two — the projection
missing a run the journal records, or carrying a stale primary_state
— is a doctor failure, not a kernel failure: the journal is
authoritative and the projection can always be rebuilt with
``multiverse rebuild-index``. The probe surfaces the gap so the
operator knows when a rebuild is owed.
"""

from __future__ import annotations

from pathlib import Path

from ..index_projection import verify_projection_against_journal
from .health_probes import (CleanupResult, LeakInventoryResult, ProbeOutcome,
                            ProbeReport)


def probe_projection_consistency(state_root: Path) -> ProbeReport:
    """Compare the journal's view of runs against the SQLite projection.

    The probe always returns a :class:`ProbeReport`. The result is:

    * ``PASS`` when projection and journal agree (or when neither has
      seen any runs yet).
    * ``FAIL`` when any drift exists. The detail string carries the
      first few offending attempt ids so the operator can investigate
      without dumping the full report into the terminal.
    * ``SKIPPED`` when the journal directory is absent — the projection
      cannot drift from a record that does not exist.

    The probe is read-only.
    """
    journal_root = Path(state_root) / "journal"
    if not journal_root.is_dir():
        return ProbeReport(
            name="index.projection_consistency",
            probe=ProbeOutcome.SKIPPED,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail=f"no journal at {str(journal_root)!r}",
        )

    try:
        report = verify_projection_against_journal(Path(state_root))
    except Exception as exc:  # noqa: BLE001 — doctor surfaces, doesn't fix
        return ProbeReport(
            name="index.projection_consistency",
            probe=ProbeOutcome.FAIL,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail=f"verification failed: {type(exc).__name__}: {exc}",
        )

    base_detail = (
        f"runs_in_journal={report.runs_in_journal}, "
        f"runs_in_projection={report.runs_in_projection}"
    )
    if report.in_sync:
        return ProbeReport(
            name="index.projection_consistency",
            probe=ProbeOutcome.PASS,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail=base_detail,
        )

    # Surface up to three offending attempts so the detail line stays
    # readable; the JSON section of `doctor --json` carries every drift
    # via ``verify_projection_against_journal`` if a caller wants more.
    sample = ", ".join(
        f"{d.physical_attempt_id[:12]}({d.kind})" for d in report.drifts[:3]
    )
    suffix = "" if len(report.drifts) <= 3 else f", +{len(report.drifts) - 3} more"
    return ProbeReport(
        name="index.projection_consistency",
        probe=ProbeOutcome.FAIL,
        cleanup=CleanupResult.CLEAN,
        leak=LeakInventoryResult.LEAKS,
        leak_count=len(report.drifts),
        detail=(
            f"{base_detail} | {len(report.drifts)} drift(s): {sample}{suffix}. "
            "Run `multiverse rebuild-index` to reconcile."
        ),
    )
