"""Doctor probe: stuck broker reservations (STRATEGY M3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.doctor import probe_reservation_ledger
from multiverse.doctor.health_probes import CleanupResult, ProbeOutcome
from multiverse.journal import JournalKind, JournalLayout, JournalWriter

pytestmark = pytest.mark.control_plane


def _write_journal(
    state_root: Path,
    records: list[tuple[JournalKind, str, dict, dict | None]],
) -> None:
    """Each tuple is ``(kind, attempt_id, payload, extra_kwargs)``."""
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test")
    for kind, attempt, payload, extra in records:
        kwargs = {"physical_attempt_id": attempt, "payload": payload}
        if extra:
            kwargs.update(extra)
        writer.append(kind, **kwargs)
    writer.commit()
    writer.close()


def test_probe_passes_when_no_journal(tmp_path: Path) -> None:
    report = probe_reservation_ledger(tmp_path)
    assert report.probe is ProbeOutcome.SKIPPED


def test_probe_passes_when_ledger_empty(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r1", {"manifest_path": "/m"}, None),
            (
                JournalKind.RESERVATION_GRANTED,
                "r1",
                {
                    "ram_bytes": 1024,
                    "vram_bytes": 0,
                    "gpu_index": None,
                    "disk_bytes_per_path": {},
                },
                None,
            ),
            (JournalKind.RESERVATION_RELEASED, "r1", {"reason": "terminal"}, None),
        ],
    )
    report = probe_reservation_ledger(tmp_path)
    assert report.probe is ProbeOutcome.PASS
    assert report.leak_count == 0


def test_probe_flags_reservation_with_terminal_run(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r-stuck", {"manifest_path": "/m"}, None),
            (
                JournalKind.RESERVATION_GRANTED,
                "r-stuck",
                {
                    "ram_bytes": 2048,
                    "vram_bytes": 0,
                    "gpu_index": None,
                    "disk_bytes_per_path": {},
                },
                None,
            ),
            (
                JournalKind.STATE_TRANSITION,
                "r-stuck",
                {"from_state": "RUNNING", "to_state": "FAILED"},
                None,
            ),
        ],
    )
    report = probe_reservation_ledger(tmp_path)
    assert report.probe is ProbeOutcome.FAIL
    assert report.leak_count == 1
    assert report.cleanup is CleanupResult.LEAKED
    assert "r-stuck" in (report.detail or "")


def test_probe_flags_orphan_reservation_with_no_job_intent(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (
                JournalKind.RESERVATION_GRANTED,
                "r-orphan",
                {
                    "ram_bytes": 1,
                    "vram_bytes": 0,
                    "gpu_index": None,
                    "disk_bytes_per_path": {},
                },
                None,
            ),
        ],
    )
    report = probe_reservation_ledger(tmp_path)
    assert report.probe is ProbeOutcome.FAIL
    assert report.leak_count == 1


def test_probe_flags_stale_reservation(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r-slow", {"manifest_path": "/m"}, None),
            (
                JournalKind.RESERVATION_GRANTED,
                "r-slow",
                {
                    "ram_bytes": 1,
                    "vram_bytes": 0,
                    "gpu_index": None,
                    "disk_bytes_per_path": {},
                },
                None,
            ),
            (
                JournalKind.STATE_TRANSITION,
                "r-slow",
                {"from_state": "PENDING", "to_state": "RUNNING"},
                None,
            ),
        ],
    )
    # With a generous stale_after, the probe should not fire.
    fresh = probe_reservation_ledger(tmp_path, stale_after_seconds=3600 * 24)
    assert fresh.probe is ProbeOutcome.PASS
    # With a zero-second threshold and a future "now", the probe fires.
    stale = probe_reservation_ledger(
        tmp_path,
        stale_after_seconds=0,
        now_iso="2099-01-01T00:00:00+00:00",
    )
    assert stale.probe is ProbeOutcome.FAIL
    assert stale.leak_count == 1


def test_probe_passes_for_active_running_reservation(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r-live", {"manifest_path": "/m"}, None),
            (
                JournalKind.RESERVATION_GRANTED,
                "r-live",
                {
                    "ram_bytes": 1,
                    "vram_bytes": 0,
                    "gpu_index": None,
                    "disk_bytes_per_path": {},
                },
                None,
            ),
            (
                JournalKind.STATE_TRANSITION,
                "r-live",
                {"from_state": "PENDING", "to_state": "RUNNING"},
                None,
            ),
        ],
    )
    # Default staleness is 30 min; a freshly-written record is well under.
    report = probe_reservation_ledger(tmp_path)
    assert report.probe is ProbeOutcome.PASS
    assert report.leak_count == 0
