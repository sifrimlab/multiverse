"""Milestone-12 exit-gate tests for manual-first GC.

Coverage:
    1. ``build_plan`` with no retention configured keeps everything.
    2. Promoted artifacts are protected by default (R12 acceptance).
    3. ``apply=False`` (default) writes a dry-run report and deletes nothing.
    4. ``apply=True`` deletes only WOULD_DELETE entries; owner-token race
       detection refuses at the last moment.
    5. ``--no-export-required`` only matters when retention has expired.
    6. Tier-1 closed list: paths enumerated verbatim; sweep removes only
       within that list and only after TTL.
    7. Tier-1 never touches user-visible directories (artifacts,
       workspaces, quarantine, cancelled, failed).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from multiverse.gc import (
    CandidateKind,
    GcCandidate,
    GcResult,
    PlanReason,
    RetentionPolicy,
    TIER1_PATHS,
    apply_plan,
    build_plan,
    enumerate_candidates,
    sweep_tier1,
)
from multiverse.gc.apply import GC_REPORTS_SUBDIR
from multiverse.promotion import StoreLayout
from multiverse.promotion.tokens import write_owner_token


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> StoreLayout:
    return StoreLayout(root=tmp_path / "store").ensure()


def _seed_failed_workspace(
    store: StoreLayout, name: str, *, age_seconds: int, with_token: bool = True,
    with_export: bool = False,
) -> Path:
    path = store.failed / name
    path.mkdir(parents=True)
    (path / "container.log").write_text("ok")
    if with_token:
        write_owner_token(
            path,
            owner_token=f"tok-{name}",
            physical_attempt_id=f"att-{name}",
            mvd_boot_id="B",
            purpose="failed-attempt",
        )
    if with_export:
        (path / "EXPORTED").write_text("yes")
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


def _seed_promoted_artifact(
    store: StoreLayout, name: str, *, age_seconds: int = 0
) -> Path:
    path = store.artifacts / name
    path.mkdir(parents=True)
    (path / "artifact_manifest.json").write_text("{}")
    write_owner_token(
        path,
        owner_token=f"tok-{name}",
        physical_attempt_id=f"att-{name}",
        mvd_boot_id="B",
        purpose="promotion-prepare",
    )
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


# ---------------------------------------------------------------------------
# 1. Default retention is infinite → keep everything
# ---------------------------------------------------------------------------


def test_default_retention_keeps_everything(store: StoreLayout) -> None:
    _seed_failed_workspace(store, "f1", age_seconds=10 * 365 * 24 * 3600)
    _seed_promoted_artifact(store, "p1", age_seconds=10 * 365 * 24 * 3600)
    plan = build_plan(enumerate_candidates(store), policy=RetentionPolicy())
    assert plan.to_delete == []
    # Two candidates, both kept.
    assert len(plan.to_keep) == 2


# ---------------------------------------------------------------------------
# 2. Promoted artifacts are protected by default (R12)
# ---------------------------------------------------------------------------


def test_promoted_artifacts_protected_by_default(store: StoreLayout) -> None:
    _seed_promoted_artifact(store, "p1", age_seconds=999 * 24 * 3600)
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(failed_workspaces_seconds=60),
    )
    [entry] = plan.entries
    assert entry.candidate.kind is CandidateKind.PROMOTED_ARTIFACT
    assert entry.reason is PlanReason.KEEP_PROMOTED_PROTECTED


def test_apply_to_promoted_required_to_consider_promoted(store: StoreLayout) -> None:
    p = _seed_promoted_artifact(store, "p1", age_seconds=200)
    # Even with apply_to_promoted=True the default RetentionPolicy has no
    # threshold for promoted → still kept (threshold None).
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(),
        apply_to_promoted=True,
    )
    [entry] = plan.entries
    assert entry.reason is PlanReason.KEEP_NO_RETENTION or \
        entry.reason is PlanReason.KEEP_PROMOTED_PROTECTED
    # And RetentionPolicy.promoted_artifacts_seconds cannot be set.
    with pytest.raises(TypeError):
        RetentionPolicy(promoted_artifacts_seconds=10)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3. Dry-run is the default; never deletes
# ---------------------------------------------------------------------------


def test_dry_run_default_never_deletes(tmp_path: Path, store: StoreLayout) -> None:
    failed = _seed_failed_workspace(
        store, "old_failed", age_seconds=8 * 24 * 3600, with_export=True
    )
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(failed_workspaces_seconds=7 * 24 * 3600),
    )
    # Plan would delete the failed one.
    assert [e.candidate.path for e in plan.to_delete] == [failed]
    result = apply_plan(plan, store_root=store.root, apply=False)
    assert isinstance(result, GcResult)
    assert result.deleted_paths == []
    # Failed dir still on disk.
    assert failed.is_dir()
    # Report path created.
    assert result.report_path is not None and result.report_path.is_file()
    assert result.report_path.parent.name == GC_REPORTS_SUBDIR


# ---------------------------------------------------------------------------
# 4. apply=True deletes only WOULD_DELETE entries
# ---------------------------------------------------------------------------


def test_apply_true_deletes_woulddelete_entries(store: StoreLayout) -> None:
    failed = _seed_failed_workspace(
        store, "f", age_seconds=10 * 24 * 3600, with_export=True
    )
    kept = _seed_failed_workspace(
        store, "k", age_seconds=10, with_export=True
    )
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(failed_workspaces_seconds=24 * 3600),
    )
    result = apply_plan(plan, store_root=store.root, apply=True)
    assert failed in result.deleted_paths
    assert kept not in result.deleted_paths
    assert not failed.exists()
    assert kept.is_dir()


def test_apply_refuses_when_owner_token_vanishes_mid_run(store: StoreLayout) -> None:
    failed = _seed_failed_workspace(
        store, "f", age_seconds=10 * 24 * 3600, with_export=True
    )
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(failed_workspaces_seconds=24 * 3600),
    )
    # Race: the token disappears between plan and apply.
    (failed / ".mvd_owner").unlink()
    result = apply_plan(plan, store_root=store.root, apply=True)
    assert failed in result.refused_paths
    assert failed.is_dir()


# ---------------------------------------------------------------------------
# 5. require_export gate
# ---------------------------------------------------------------------------


def test_no_export_keeps_the_candidate(store: StoreLayout) -> None:
    no_export = _seed_failed_workspace(
        store, "no_export", age_seconds=30 * 24 * 3600, with_export=False
    )
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(failed_workspaces_seconds=7 * 24 * 3600),
        require_export=True,
    )
    [entry] = plan.entries
    assert entry.reason is PlanReason.KEEP_NO_EXPORT
    assert no_export.is_dir()


def test_require_export_false_allows_deletion_when_aged(store: StoreLayout) -> None:
    aged = _seed_failed_workspace(
        store, "a", age_seconds=30 * 24 * 3600, with_export=False
    )
    plan = build_plan(
        enumerate_candidates(store),
        policy=RetentionPolicy(failed_workspaces_seconds=7 * 24 * 3600),
        require_export=False,
    )
    [entry] = plan.entries
    assert entry.reason is PlanReason.WOULD_DELETE
    apply_plan(plan, store_root=store.root, apply=True)
    assert not aged.exists()


# ---------------------------------------------------------------------------
# 6. Tier-1 closed list
# ---------------------------------------------------------------------------


def test_tier1_paths_are_a_closed_list() -> None:
    """Adding a Tier-1 path requires editing this test. The list lives in
    multiverse/gc/tier1.py — keep them in sync."""
    expected = {
        ("state", "journal/rotated", 90 * 24 * 3600),
        ("store", "workspaces/__mvd_health_probe__", 3600),
    }
    assert set(TIER1_PATHS) == expected


def test_tier1_removes_only_expired_rotated_segments(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    store_root = tmp_path / "store"
    rotated = state_root / "journal" / "rotated"
    rotated.mkdir(parents=True)
    fresh = rotated / "fresh.log"
    fresh.write_bytes(b"x")
    stale = rotated / "stale.log"
    stale.write_bytes(b"y")
    old = time.time() - 91 * 24 * 3600
    os.utime(stale, (old, old))

    result = sweep_tier1(state_root=state_root, store_root=store_root)
    assert result.removed_per_path["journal/rotated"] == 1
    assert fresh.exists()
    assert not stale.exists()


def test_tier1_never_touches_user_visible_directories(tmp_path: Path) -> None:
    """Cardinal R12 invariant: even if a user-visible directory is very
    old, Tier-1 sweep refuses to look at it."""
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = StoreLayout(root=tmp_path / "store").ensure()
    # Seed an ancient promoted artifact AND an ancient failed workspace.
    promoted = _seed_promoted_artifact(store, "ancient_promoted", age_seconds=10_000_000)
    failed = _seed_failed_workspace(store, "ancient_failed", age_seconds=10_000_000)
    quarantine_dir = store.quarantine / "2020-01-01" / "ancient_q"
    quarantine_dir.mkdir(parents=True)
    (quarantine_dir / "marker").write_bytes(b"x")
    old = time.time() - 10_000_000
    os.utime(quarantine_dir, (old, old))

    sweep_tier1(state_root=state_root, store_root=store.root)

    # All user-visible dirs intact.
    assert promoted.is_dir()
    assert failed.is_dir()
    assert quarantine_dir.is_dir()


# ---------------------------------------------------------------------------
# 7. Closed-list defence: no path under store/artifacts/ in Tier-1
# ---------------------------------------------------------------------------


def test_tier1_closed_list_excludes_user_visible_namespaces() -> None:
    forbidden_substrings = ("artifacts", "quarantine", "cancelled", "failed")
    for root_kind, rel, _ttl in TIER1_PATHS:
        for token in forbidden_substrings:
            assert token not in rel, (
                f"Tier-1 path {rel!r} contains forbidden substring {token!r}"
            )
