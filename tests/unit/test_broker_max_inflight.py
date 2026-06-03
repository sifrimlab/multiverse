"""Broker max-inflight admission predicate (STRATEGY M4 §2).

When ``max_inflight_dispatches`` is set, the broker stops doing RAM
math and instead enforces a dispatch budget: admit until ``inflight ==
max_inflight``, then refuse. Pressure-mode and GPU-serialization checks
do not fire under this policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.broker import (AdmissionOutcome, HostMetrics,
                               InMemoryHostObserver, ResourceBroker,
                               ResourceRequest)
from multiverse.journal import (JournalKind, JournalLayout, JournalReader,
                                JournalWriter)

pytestmark = pytest.mark.control_plane


def _metrics() -> HostMetrics:
    # Deliberately tight RAM: under the local policy this would refuse
    # the second admission, but under max-inflight it is irrelevant.
    return HostMetrics(ram_free_bytes=128, ram_total_bytes=1024)


def _broker(*, max_inflight: int = 8) -> ResourceBroker:
    return ResourceBroker(
        observer=InMemoryHostObserver(_metrics()),
        max_inflight_dispatches=max_inflight,
    )


def test_admit_until_budget_hit() -> None:
    broker = _broker(max_inflight=3)
    for i in range(3):
        decision = broker.admit(
            physical_attempt_id=f"r{i}",
            request=ResourceRequest(ram_bytes=1),
        )
        assert decision.admitted, decision
    full = broker.admit(
        physical_attempt_id="r3",
        request=ResourceRequest(ram_bytes=1),
    )
    assert not full.admitted
    assert full.outcome is AdmissionOutcome.REJECTED_INSUFFICIENT
    assert "max_inflight=3" in (full.detail or "")


def test_release_frees_a_slot() -> None:
    broker = _broker(max_inflight=1)
    a = broker.admit(physical_attempt_id="r1", request=ResourceRequest(ram_bytes=1))
    assert a.admitted
    refused = broker.admit(
        physical_attempt_id="r2", request=ResourceRequest(ram_bytes=1)
    )
    assert not refused.admitted
    broker.release("r1")
    again = broker.admit(physical_attempt_id="r2", request=ResourceRequest(ram_bytes=1))
    assert again.admitted


def test_max_inflight_ignores_ram_math() -> None:
    """Under the inflight policy the broker must not compare RAM
    against the live observer — only the dispatch count matters."""
    broker = _broker(max_inflight=10)
    # ram_free = 128 bytes; request 1 GiB. Local policy would refuse.
    decision = broker.admit(
        physical_attempt_id="big",
        request=ResourceRequest(ram_bytes=1 << 30),
    )
    assert decision.admitted


def test_pressure_thresholds_do_not_fire_under_inflight_policy() -> None:
    # Force a tight observer that local policy would flag as CRITICAL.
    crit_metrics = HostMetrics(ram_free_bytes=1, ram_total_bytes=1024)
    broker = ResourceBroker(
        observer=InMemoryHostObserver(crit_metrics),
        max_inflight_dispatches=2,
    )
    decision = broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=1),
    )
    assert decision.admitted


def test_hpo_flood_of_50_respects_budget(tmp_path: Path) -> None:
    """STRATEGY M4 acceptance: an HPO sweep of 50 trials respects
    max_inflight and never grants more than ``budget`` reservations at
    once."""
    layout = JournalLayout.at(tmp_path / "journal").ensure()
    journal = JournalWriter(layout, boot_id="boot-test")
    broker = ResourceBroker(
        observer=InMemoryHostObserver(_metrics()),
        max_inflight_dispatches=8,
        journal=journal,
    )

    admitted: list[str] = []
    refused: list[str] = []
    for i in range(50):
        attempt = f"r{i:02d}"
        d = broker.admit(
            physical_attempt_id=attempt,
            request=ResourceRequest(ram_bytes=1),
        )
        if d.admitted:
            admitted.append(attempt)
        else:
            refused.append(attempt)

    assert len(admitted) == 8
    assert len(refused) == 42
    # Drain a couple, then resubmit those refused — they must now flow.
    for a in admitted[:3]:
        broker.release(a)
    again_admitted = 0
    for a in refused[:3]:
        d = broker.admit(
            physical_attempt_id=a + "-retry",
            request=ResourceRequest(ram_bytes=1),
        )
        if d.admitted:
            again_admitted += 1
    journal.close()
    assert again_admitted == 3

    # And the journal carries grant/release records for every admission.
    records = JournalReader(layout).replay().records
    grants = [r for r in records if r.kind is JournalKind.RESERVATION_GRANTED]
    assert len(grants) == 8 + 3  # original cohort + 3 retries
