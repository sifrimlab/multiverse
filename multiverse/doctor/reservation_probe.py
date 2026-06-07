"""Doctor probe: stuck broker reservations (STRATEGY M3).

The broker's reservation ledger is durable: ``RESERVATION_GRANTED`` and
``RESERVATION_RELEASED`` records in the journal are the source of truth.
A reservation is *stuck* if it was granted, never released, and the run
it belongs to is either:

* in a terminal state (so the executor crashed mid-finally), or
* missing from the journal entirely (corruption / partial truncate), or
* in a non-terminal state but the grant is older than a configurable
  staleness threshold (the kernel hung).

The probe reports per-attempt detail so the operator can either resume
the kernel (which will release on its next replay) or reissue the
attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..broker import reconstruct_ledger_from_journal
from ..journal import JournalKind, JournalLayout, JournalReader, JournalRecord
from .health_probes import (CleanupResult, LeakInventoryResult, ProbeOutcome,
                            ProbeReport)

DEFAULT_STALE_AFTER_SECONDS = 30 * 60  # 30 minutes


@dataclass(frozen=True)
class StuckReservation:
    """A broker lease that was granted but never released.

    Attributes:
        physical_attempt_id: The attempt that holds the unreleased grant.
        ram_bytes: RAM the lease reserved (so the operator sees what is
            being held out of the broker's pool).
        granted_wall_iso: Wall-clock ISO timestamp of the grant, or
            ``"unknown"`` if the grant record could not be located.
        terminal_state: Final run state when the leak is attributable to a
            terminal run; ``None`` for unknown/stale reasons.
        reason: Why the lease is stuck — one of ``"terminal"``,
            ``"unknown_run"``, or ``"stale"``.
    """

    physical_attempt_id: str
    ram_bytes: int
    granted_wall_iso: str
    terminal_state: Optional[str]
    """``None`` if the run is unknown to the journal (no JOB_INTENT seen)."""
    reason: str

    def to_dict(self) -> Dict[str, object]:
        """Render as a JSON-serialisable row for the doctor report."""
        return {
            "physical_attempt_id": self.physical_attempt_id,
            "ram_bytes": int(self.ram_bytes),
            "granted_wall_iso": self.granted_wall_iso,
            "terminal_state": self.terminal_state,
            "reason": self.reason,
        }


def probe_reservation_ledger(
    state_root: Path,
    *,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    now_iso: Optional[str] = None,
) -> ProbeReport:
    """Scan the journal and report leases granted but never released.

    Read-only. Reconstructs the broker ledger from the journal, then for
    each in-flight grant classifies it as stuck when the run is terminal,
    unknown to the journal, or non-terminal but older than the staleness
    threshold (the kernel hung).

    Args:
        state_root: Root of the mvd state tree; the journal is read from
            ``state_root/journal``.
        stale_after_seconds: Age past which a still-running grant is
            treated as stale (the kernel is presumed hung).
        now_iso: Reference "now" as an ISO timestamp; defaults to the
            current UTC time. Injected by tests for deterministic ages.

    Returns:
        A :class:`ProbeReport` named ``broker.reservation_ledger``:
        ``SKIPPED`` when no journal exists, ``PASS`` when nothing is stuck,
        ``FAIL`` (with a leak count) otherwise.
    """
    journal_root = Path(state_root) / "journal"
    if not journal_root.is_dir():
        return ProbeReport(
            name="broker.reservation_ledger",
            probe=ProbeOutcome.SKIPPED,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail=f"no journal at {str(journal_root)!r}",
        )

    try:
        records = JournalReader(JournalLayout.at(journal_root)).replay().records
    except Exception as exc:  # noqa: BLE001 — doctor surfaces, doesn't fix
        return ProbeReport(
            name="broker.reservation_ledger",
            probe=ProbeOutcome.FAIL,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail=f"journal replay failed: {type(exc).__name__}: {exc}",
        )

    ledger = reconstruct_ledger_from_journal(records)
    if not ledger.by_attempt:
        return ProbeReport(
            name="broker.reservation_ledger",
            probe=ProbeOutcome.PASS,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail="ledger empty",
        )

    grant_by_attempt = _latest_grant_by_attempt(records)
    terminal_by_attempt = _terminal_state_by_attempt(records)
    now = _parse_iso(now_iso) if now_iso else datetime.now(timezone.utc)

    stuck: List[StuckReservation] = []
    for attempt, request in ledger.by_attempt.items():
        grant_record = grant_by_attempt.get(attempt)
        grant_iso = grant_record.wall_iso if grant_record else "unknown"
        terminal_state = terminal_by_attempt.get(attempt)
        if attempt not in _seen_job_intent(records):
            stuck.append(
                StuckReservation(
                    physical_attempt_id=attempt,
                    ram_bytes=request.ram_bytes,
                    granted_wall_iso=grant_iso,
                    terminal_state=None,
                    reason="unknown_run",
                )
            )
            continue
        if terminal_state is not None:
            stuck.append(
                StuckReservation(
                    physical_attempt_id=attempt,
                    ram_bytes=request.ram_bytes,
                    granted_wall_iso=grant_iso,
                    terminal_state=terminal_state,
                    reason="terminal",
                )
            )
            continue
        if (
            grant_record is not None
            and _age_seconds(grant_record, now) >= stale_after_seconds
        ):
            stuck.append(
                StuckReservation(
                    physical_attempt_id=attempt,
                    ram_bytes=request.ram_bytes,
                    granted_wall_iso=grant_iso,
                    terminal_state=None,
                    reason="stale",
                )
            )

    if not stuck:
        return ProbeReport(
            name="broker.reservation_ledger",
            probe=ProbeOutcome.PASS,
            cleanup=CleanupResult.CLEAN,
            leak=LeakInventoryResult.NONE,
            leak_count=0,
            detail=f"in_flight={len(ledger.by_attempt)}; none stuck",
        )

    pieces = [f"{s.physical_attempt_id[:12]}({s.reason})" for s in stuck]
    return ProbeReport(
        name="broker.reservation_ledger",
        probe=ProbeOutcome.FAIL,
        cleanup=CleanupResult.LEAKED,
        leak=LeakInventoryResult.LEAKS,
        leak_count=len(stuck),
        detail=(
            f"stuck reservations: {', '.join(pieces)}. "
            "Restart the kernel to trigger replay-time release, or "
            "inspect the journal for the responsible run."
        ),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _latest_grant_by_attempt(
    records: List[JournalRecord],
) -> Dict[str, JournalRecord]:
    """Map each attempt to its most recent RESERVATION_GRANTED record."""
    out: Dict[str, JournalRecord] = {}
    for record in records:
        if (
            record.kind is JournalKind.RESERVATION_GRANTED
            and record.physical_attempt_id
        ):
            out[record.physical_attempt_id] = record
    return out


def _terminal_state_by_attempt(
    records: List[JournalRecord],
) -> Dict[str, str]:
    """Map each attempt to its terminal state, if it reached one.

    An attempt that later transitions back out of a terminal state is
    dropped from the map (rare, but treated as not-terminal).
    """
    # Import locally to avoid circular: doctor → mvd → doctor.
    from ..mvd.state import PrimaryState

    out: Dict[str, str] = {}
    for record in records:
        if record.kind is not JournalKind.STATE_TRANSITION:
            continue
        attempt = record.physical_attempt_id
        next_state = record.payload.get("to_state")
        if not attempt or not next_state:
            continue
        try:
            state = PrimaryState(next_state)
        except ValueError:
            continue
        if state.is_terminal:
            out[attempt] = state.value
        elif attempt in out:
            # State moved out of terminal — rare, but treat as not-terminal.
            del out[attempt]
    return out


def _seen_job_intent(records: List[JournalRecord]) -> set[str]:
    """Return attempt ids that have a JOB_INTENT record (known runs)."""
    return {
        record.physical_attempt_id
        for record in records
        if record.kind is JournalKind.JOB_INTENT and record.physical_attempt_id
    }


def _parse_iso(value: str) -> datetime:
    """Parse an ISO timestamp, falling back to current UTC on garbage."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)


def _age_seconds(record: JournalRecord, now: datetime) -> float:
    """Seconds between a record's wall time and ``now`` (>= 0; naive
    timestamps are treated as UTC)."""
    try:
        granted = datetime.fromisoformat(record.wall_iso)
    except ValueError:
        return 0.0
    if granted.tzinfo is None:
        granted = granted.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - granted).total_seconds())
