"""Milestone-11 exit-gate tests for doctor + storage probes.

Coverage:
    1. Storage probes on a healthy tmp path → ``supported``/``degraded``
       per-probe matrix, ``can_start`` returns True.
    2. Cloud-sync marker detection → ``dangerous`` for that probe and
       refusal-to-start without ``--accept-degraded``.
    3. Read-only root → ``write_then_read`` returns ``blocked``.
    4. Free-space tiers map correctly to supported/degraded/dangerous.
    5. Health probes write inside reserved namespace only; probe report
       has the three columns from R9 (probe/cleanup/leak).
    6. Sweeper removes only entries older than the TTL, never touches
       user-visible directories.
    7. Doctor report aggregates section statuses (OK/WARNING/BLOCKED).
    8. Reserved namespaces match R9 verbatim.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from multiverse.doctor import (BLOCKED, DANGEROUS, DEGRADED,
                               HEALTH_PROBE_NAMESPACES,
                               HEALTH_PROBE_TTL_SECONDS, SUPPORTED,
                               DoctorReport, DoctorSection, ProbeOutcome,
                               SectionStatus, sweep_expired_health_probes)
from multiverse.doctor.health_probes import (CleanupResult,
                                             LeakInventoryResult,
                                             probe_workspace_directory)
from multiverse.doctor.storage_probes import (StorageLevel,
                                              probe_atomic_rename,
                                              probe_cloud_sync_heuristic,
                                              probe_free_space,
                                              probe_fsync_dir,
                                              probe_fsync_file,
                                              probe_write_then_read,
                                              run_storage_probes)

# ---------------------------------------------------------------------------
# 1. Storage probes on a healthy path
# ---------------------------------------------------------------------------


def test_storage_probes_healthy_path(tmp_path: Path) -> None:
    report = run_storage_probes(tmp_path)
    by_name = report.by_name()
    # Required-supported probes must report SUPPORTED on a normal Linux fs.
    assert by_name["write_then_read"].level is SUPPORTED
    assert by_name["atomic_rename"].level is SUPPORTED
    assert by_name["fsync_file"].level is SUPPORTED
    # Cloud-sync should be clean on an isolated tmp_path.
    assert by_name["cloud_sync_heuristic"].level is SUPPORTED
    # can_start under normal conditions.
    assert report.can_start() is True
    # Probe dir cleaned up.
    assert not (tmp_path / "_probe").exists()


# ---------------------------------------------------------------------------
# 2. Cloud-sync detection
# ---------------------------------------------------------------------------


def test_cloud_sync_marker_yields_dangerous(tmp_path: Path) -> None:
    # Drop a Dropbox marker beside the candidate root.
    (tmp_path / ".dropbox").mkdir()
    root = tmp_path / "store"
    root.mkdir()
    result = probe_cloud_sync_heuristic(root)
    assert result.level is DANGEROUS
    assert (
        "cloud-sync markers" in (result.detail or "").lower()
        or "cloud-sync" in (result.detail or "").lower()
    )


def test_path_name_marker_yields_dangerous(tmp_path: Path) -> None:
    onedrive = tmp_path / "OneDrive" / "store"
    onedrive.mkdir(parents=True)
    result = probe_cloud_sync_heuristic(onedrive)
    assert result.level is DANGEROUS


def test_dangerous_requires_accept_degraded_to_start(tmp_path: Path) -> None:
    (tmp_path / ".dropbox").mkdir()
    root = tmp_path / "store"
    root.mkdir()
    report = run_storage_probes(root)
    # Dangerous, so can_start is False unless explicitly accepted.
    assert report.can_start(accept_degraded=False) is False
    assert report.can_start(accept_degraded=True) is True


# ---------------------------------------------------------------------------
# 3. Read-only root → blocked
# ---------------------------------------------------------------------------


def test_write_then_read_blocked_on_readonly_root(tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o555)
    try:
        result = probe_write_then_read(ro)
        # On some filesystems chmod 0o555 still permits owner write (root
        # in some CI envs). Accept either BLOCKED or SUPPORTED-not-blocked,
        # but require the report-level can_start logic when blocked.
        if result.level is BLOCKED:
            report = run_storage_probes(ro)
            assert report.can_start() is False
        else:
            # The probe could not detect blocked status in this env; skip
            # the rest. Don't fail; the invariant is "if blocked-probe,
            # then can_start False".
            pytest.skip("chmod 0o555 did not block writes in this environment")
    finally:
        os.chmod(ro, 0o755)


# ---------------------------------------------------------------------------
# 4. Free-space tiers
# ---------------------------------------------------------------------------


def test_free_space_tiers_map_to_levels(tmp_path: Path) -> None:
    # Tune thresholds so the test deterministically lands on each level.
    # Use absurdly small SUPPORTED threshold to ensure SUPPORTED on tmp.
    supported = probe_free_space(
        tmp_path, min_supported_gb=0.0001, min_degraded_gb=0.00001
    )
    assert supported.level is SUPPORTED

    # Now an absurdly large threshold — tmp_path cannot have terabytes free.
    dangerous = probe_free_space(tmp_path, min_supported_gb=1e9, min_degraded_gb=1e8)
    assert dangerous.level in (DEGRADED, DANGEROUS)


# ---------------------------------------------------------------------------
# 5. Health probes use reserved namespaces only
# ---------------------------------------------------------------------------


def test_workspace_health_probe_uses_reserved_namespace(tmp_path: Path) -> None:
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    report = probe_workspace_directory(workspaces)
    assert report.name == "workspace_dir"
    assert report.probe is ProbeOutcome.PASS
    assert report.cleanup is CleanupResult.CLEAN
    assert report.leak is LeakInventoryResult.NONE
    # Probes write only inside the reserved namespace.
    children = sorted(p.name for p in workspaces.iterdir())
    assert children == [HEALTH_PROBE_NAMESPACES["workspace_dir"]]
    # And the probe dir is empty after a clean cycle.
    reserved = workspaces / HEALTH_PROBE_NAMESPACES["workspace_dir"]
    assert list(reserved.iterdir()) == []


def test_workspace_health_probe_three_columns_when_leaks_present(
    tmp_path: Path,
) -> None:
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    reserved = workspaces / HEALTH_PROBE_NAMESPACES["workspace_dir"]
    reserved.mkdir()
    # Plant a stale entry — older than the TTL.
    stale = reserved / "probe-leak"
    stale.mkdir()
    old = time.time() - HEALTH_PROBE_TTL_SECONDS - 10
    os.utime(stale, (old, old))

    report = probe_workspace_directory(workspaces)
    # The probe itself still passes and cleans its own entry — leaks are
    # an *independent* column.
    assert report.probe is ProbeOutcome.PASS
    assert report.cleanup is CleanupResult.CLEAN
    assert report.leak is LeakInventoryResult.LEAKS
    assert report.leak_count >= 1


# ---------------------------------------------------------------------------
# 6. Sweeper TTL behaviour
# ---------------------------------------------------------------------------


def test_sweeper_removes_only_expired_entries(tmp_path: Path) -> None:
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    reserved = workspaces / HEALTH_PROBE_NAMESPACES["workspace_dir"]
    reserved.mkdir()

    fresh = reserved / "fresh"
    fresh.mkdir()
    stale = reserved / "stale"
    stale.mkdir()
    (stale / "marker").write_bytes(b"x")
    old = time.time() - HEALTH_PROBE_TTL_SECONDS - 100
    os.utime(stale, (old, old))

    removed = sweep_expired_health_probes(workspaces)
    assert removed["workspace_dir"] == 1
    assert fresh.exists()
    assert not stale.exists()


def test_sweeper_never_touches_user_visible_directories(tmp_path: Path) -> None:
    """Crucial safety invariant: even with very-old entries placed
    *outside* the reserved namespace, the sweeper leaves them alone."""
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    # A user-owned workspace, very old.
    user_ws = workspaces / "user_run_42"
    user_ws.mkdir()
    (user_ws / "container.log").write_bytes(b"important")
    old = time.time() - HEALTH_PROBE_TTL_SECONDS - 10_000
    os.utime(user_ws, (old, old))

    sweep_expired_health_probes(workspaces)
    assert user_ws.is_dir()
    assert (user_ws / "container.log").is_file()


# ---------------------------------------------------------------------------
# 7. Doctor report aggregation
# ---------------------------------------------------------------------------


def test_doctor_report_overall_status_aggregates() -> None:
    report = DoctorReport(
        sections=[
            DoctorSection(name="storage", status=SectionStatus.OK),
            DoctorSection(name="docker", status=SectionStatus.WARNING),
        ]
    )
    assert report.overall_status is SectionStatus.WARNING
    report.sections.append(DoctorSection(name="db", status=SectionStatus.BLOCKED))
    assert report.overall_status is SectionStatus.BLOCKED


# ---------------------------------------------------------------------------
# 8. Reserved namespaces match R9 verbatim
# ---------------------------------------------------------------------------


def test_health_probe_namespaces_match_strategy() -> None:
    # Per STRATEGY R9 the workspace-probe directory is
    # `store/workspaces/__mvd_health_probe__/`.
    assert HEALTH_PROBE_NAMESPACES["workspace_dir"] == "__mvd_health_probe__"
    assert HEALTH_PROBE_NAMESPACES["mlflow_experiment"] == "__mvd_health_probe__"
    assert HEALTH_PROBE_NAMESPACES["docker_label"] == "multiverse.health_probe"
    assert HEALTH_PROBE_TTL_SECONDS == 3600  # 1 h per R9


# ---------------------------------------------------------------------------
# 9. Storage report degraded_capabilities surfaces the names
# ---------------------------------------------------------------------------


def test_storage_report_lists_degraded_capabilities(tmp_path: Path) -> None:
    # Force a degraded result on at least one probe by passing absurd
    # tight free-space thresholds.
    from multiverse.doctor.storage_probes import (StorageProbeResult,
                                                  StorageReport)

    report = StorageReport(
        root=tmp_path,
        results=[
            StorageProbeResult("a", SUPPORTED),
            StorageProbeResult("b", DEGRADED),
            StorageProbeResult("c", DEGRADED),
        ],
    )
    assert sorted(report.degraded_capabilities) == ["b", "c"]
    assert report.worst_level is DEGRADED
    assert report.can_start() is True
