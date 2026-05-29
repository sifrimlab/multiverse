"""Deep Slurm doctor probe (STRATEGY M4 §5)."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import pytest

from multiverse.doctor import slurm_probe
from multiverse.doctor.health_probes import ProbeOutcome


pytestmark = pytest.mark.control_plane


def test_probe_skipped_when_sbatch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slurm_probe.shutil, "which", lambda _name: None)
    report = slurm_probe.probe_slurm_deep(smoke_test=False)
    assert report.probe is ProbeOutcome.SKIPPED


def test_probe_fails_when_sbatch_present_but_sacct_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        slurm_probe.shutil,
        "which",
        lambda name: "/usr/bin/sbatch" if name == "sbatch" else None,
    )
    report = slurm_probe.probe_slurm_deep(smoke_test=False)
    assert report.probe is ProbeOutcome.FAIL
    assert "sacct" in (report.detail or "")


def test_probe_passes_with_all_bins_no_smoke_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        slurm_probe.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    def fake_run(cmd, **kwargs):
        # sinfo enumeration
        if cmd[0] == "sinfo":
            return _Completed(0, "cpu\ngpu*\nbig-mem\n", "")
        raise AssertionError(f"unexpected subprocess: {cmd}")

    monkeypatch.setattr(slurm_probe.subprocess, "run", fake_run)
    report = slurm_probe.probe_slurm_deep(smoke_test=False)
    assert report.probe is ProbeOutcome.PASS
    assert "partitions=cpu,gpu,big-mem" in (report.detail or "")


def test_smoke_test_round_trips_a_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        slurm_probe.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    calls: list[Any] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "sinfo":
            return _Completed(0, "cpu*\n", "")
        if cmd[0] == "sbatch":
            return _Completed(0, "42\n", "")
        if cmd[0] == "sacct":
            # First poll: PENDING; second: COMPLETED.
            n_sacct = sum(1 for c in calls if c[0] == "sacct")
            return _Completed(0, "PENDING\n" if n_sacct == 1 else "COMPLETED\n", "")
        if cmd[0] == "scancel":
            return _Completed(0, "", "")
        raise AssertionError(f"unexpected subprocess: {cmd}")

    # Time.sleep is the only thing we actually want fast in this unit test.
    monkeypatch.setattr(slurm_probe.time, "sleep", lambda _s: None)
    monkeypatch.setattr(slurm_probe.subprocess, "run", fake_run)
    report = slurm_probe.probe_slurm_deep(
        smoke_test=True, smoke_timeout_seconds=5
    )
    assert report.probe is ProbeOutcome.PASS
    assert "smoke_job=42=COMPLETED" in (report.detail or "")
    # The smoke completed naturally, so no scancel was issued.
    assert not any(c[0] == "scancel" for c in calls)


def test_smoke_test_reaps_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        slurm_probe.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    calls: list[Any] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "sinfo":
            return _Completed(0, "cpu*\n", "")
        if cmd[0] == "sbatch":
            return _Completed(0, "99\n", "")
        if cmd[0] == "sacct":
            return _Completed(0, "PENDING\n", "")
        if cmd[0] == "scancel":
            return _Completed(0, "", "")
        raise AssertionError(f"unexpected subprocess: {cmd}")

    # Force the polling loop's monotonic clock to instantly cross the
    # deadline so the timeout branch fires deterministically.
    monkeypatch.setattr(slurm_probe.time, "sleep", lambda _s: None)
    times = iter([0.0, 999.0, 999.0])
    monkeypatch.setattr(slurm_probe.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(slurm_probe.subprocess, "run", fake_run)
    report = slurm_probe.probe_slurm_deep(
        smoke_test=True, smoke_timeout_seconds=1
    )
    assert report.probe is ProbeOutcome.FAIL
    assert any(c[0] == "scancel" for c in calls), (
        "probe must reap the stranded smoke job"
    )


def test_partitions_strip_default_star(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sinfo --format=%P`` appends ``*`` to the default partition; the
    probe must strip it so callers see bare partition names."""
    monkeypatch.setattr(
        slurm_probe.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    def fake_run(cmd, **kwargs):
        return _Completed(0, "cpu*\ngpu\n", "")

    monkeypatch.setattr(slurm_probe.subprocess, "run", fake_run)
    partitions = slurm_probe._enumerate_partitions(timeout_seconds=5)
    assert partitions == ["cpu", "gpu"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Completed:
    """Tiny stand-in for subprocess.CompletedProcess that ignores
    keyword args (timeout, check, capture_output, text)."""

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
