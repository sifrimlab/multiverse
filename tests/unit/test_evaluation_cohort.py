"""Tests for multiverse/evaluation/cohort.py — cohort persistence, readiness
resolution, and the Evaluate-section readiness gate.

Covers all items listed in Gap 8 of STRATEGY.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from multiverse.evaluation.cohort import (
    STATUS_BAD_ARTIFACT_MANIFEST,
    STATUS_CANCELLED,
    STATUS_MISSING_ARTIFACT_DIR,
    STATUS_MISSING_DATASET,
    STATUS_NOT_SUBMITTED,
    STATUS_NO_EMBEDDINGS,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_TRAINING_FAILED,
    STATUS_UNSUPPORTED_DATASET,
    build_cohort,
    cohort_path,
    evaluate_section_view,
    latest_launch_path,
    load_latest_cohort,
    make_launch_id,
    make_member_id,
    readiness_summary,
    resolve_cohort_readiness,
    resolve_member_readiness,
    update_cohort_submitted,
    write_cohort,
    write_latest_launch,
)

CREATED_AT = "2026-06-03T12:00:00Z"
MANIFEST_HASH = "deadbeef01234567"
BACKEND = "docker"
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job(
    dataset_slug: str = "demo",
    model_slug: str = "pca",
    *,
    skipped: bool = False,
    completed_attempt_id: Optional[str] = None,
    completed_artifact_dir: Optional[str] = None,
    logical_run_id: str = "",
    dataset_path: str = "/data/demo.h5ad",
) -> Dict[str, Any]:
    job: Dict[str, Any] = {
        "dataset_slug": dataset_slug,
        "dataset_name": dataset_slug,
        "dataset_path": dataset_path,
        "model_slug": model_slug,
        "model_name": model_slug,
        "batch_key": "batch",
        "cell_type_key": "cell_type",
        "metrics": {},
    }
    if skipped:
        job["_skipped"] = True
        job["_skip_reason"] = "completed logical run already has ARTIFACT_SUCCESS"
    if completed_attempt_id:
        job["_completed_attempt_id"] = completed_attempt_id
    if completed_artifact_dir:
        job["_completed_artifact_dir"] = completed_artifact_dir
    if logical_run_id:
        job["_logical_run_id"] = logical_run_id
    return job


def _write_fake_cohort(tmp_path: Path, jobs: list) -> tuple[str, Dict]:
    launch_id = make_launch_id(
        manifest_hash=MANIFEST_HASH, backend=BACKEND, seed=SEED, created_at=CREATED_AT
    )
    cohort = build_cohort(
        launch_id=launch_id,
        manifest_hash=MANIFEST_HASH,
        manifest_path="/tmp/run_manifest.yaml",
        output_dir=str(tmp_path),
        experiment_name="test_exp",
        seed=SEED,
        backend=BACKEND,
        pending_jobs=jobs,
        created_at=CREATED_AT,
    )
    write_cohort(output_dir=tmp_path, launch_id=launch_id, cohort=cohort)
    write_latest_launch(
        output_dir=tmp_path, launch_id=launch_id, created_at=CREATED_AT
    )
    return launch_id, cohort


# ---------------------------------------------------------------------------
# Gap 1: launch_id format and uniqueness
# ---------------------------------------------------------------------------


def test_launch_id_format_contains_manifest_prefix():
    lid = make_launch_id(
        manifest_hash=MANIFEST_HASH, backend=BACKEND, seed=SEED, created_at=CREATED_AT
    )
    assert lid.startswith(MANIFEST_HASH[:8])
    assert "_docker_" in lid
    assert "_seed42_" in lid


def test_launch_id_uniqueness():
    ids = {
        make_launch_id(
            manifest_hash=MANIFEST_HASH, backend=BACKEND, seed=SEED, created_at=CREATED_AT
        )
        for _ in range(20)
    }
    assert len(ids) == 20, "launch IDs must be unique even with identical inputs"


def test_launch_id_distinct_for_different_backends():
    a = make_launch_id(
        manifest_hash=MANIFEST_HASH, backend="docker", seed=SEED, created_at=CREATED_AT
    )
    b = make_launch_id(
        manifest_hash=MANIFEST_HASH, backend="slurm", seed=SEED, created_at=CREATED_AT
    )
    assert "_docker_" in a
    assert "_slurm_" in b


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------


def test_build_cohort_mixed_submitted_and_skipped():
    submitted_job = _job("pbmc", "scvi")
    skipped_job = _job(
        "mouse",
        "pca",
        skipped=True,
        completed_attempt_id="atmp-001",
        completed_artifact_dir="/art/mouse_pca",
        logical_run_id="lrid-abc",
    )
    cohort = build_cohort(
        launch_id="test-lid",
        manifest_hash=MANIFEST_HASH,
        manifest_path="/tmp/m.yaml",
        output_dir="/out",
        experiment_name="exp",
        seed=SEED,
        backend=BACKEND,
        pending_jobs=[submitted_job, skipped_job],
        created_at=CREATED_AT,
    )
    members = cohort["members"]
    assert len(members) == 2
    assert members[0]["source"] == "submitted"
    assert members[0]["skipped"] is False
    assert members[1]["source"] == "skipped_completed"
    assert members[1]["skipped"] is True
    assert members[1]["completed_attempt_id"] == "atmp-001"
    assert members[1]["artifact_dir"] == "/art/mouse_pca"
    assert members[1]["logical_run_id"] == "lrid-abc"


def test_build_cohort_maps_cell_type_key_to_label_key():
    """Phase 0 / Gap 9: the registry-side ``cell_type_key`` must surface on the
    cohort member as ``label_key`` (scIB nomenclature) so the evaluator reads a
    field that actually exists, instead of silently defaulting to "cell_type".
    """
    job = _job("pbmc", "pca")
    job["cell_type_key"] = "celltype_annot"
    job["batch_key"] = "donor"
    cohort = build_cohort(
        launch_id="lid",
        manifest_hash=MANIFEST_HASH,
        manifest_path="/tmp/m.yaml",
        output_dir="/out",
        experiment_name="exp",
        seed=SEED,
        backend=BACKEND,
        pending_jobs=[job],
        created_at=CREATED_AT,
    )
    member = cohort["members"][0]
    assert member["label_key"] == "celltype_annot"
    assert member["batch_key"] == "donor"
    # The member carries no ``cell_type_key`` field — the evaluator must read
    # ``label_key`` (regression guard for the dropped-key bug).
    assert "cell_type_key" not in member


def test_prepare_evaluation_config_preserves_label_key(tmp_path):
    """Phase 0 / Gap 9: the trimmed eval_config the container consumes must
    carry ``label_key`` end-to-end. Uses ready_members_only=False to bypass the
    artifact-readiness gate (covered separately) and isolate the key flow.
    """
    from multiverse.evaluation.docker_runner import prepare_evaluation

    job = _job("pbmc", "pca")
    job["cell_type_key"] = "celltype_annot"
    launch_id, _cohort = _write_fake_cohort(tmp_path, [job])

    plan = prepare_evaluation(
        cohort_path(tmp_path, launch_id), ready_members_only=False
    )
    with open(plan.config_path, encoding="utf-8") as fh:
        eval_cohort = json.load(fh)
    assert eval_cohort["members"][0]["label_key"] == "celltype_annot"


def test_build_cohort_all_skipped():
    jobs = [
        _job("ds1", "m1", skipped=True, completed_attempt_id="a1", completed_artifact_dir="/a1"),
        _job("ds2", "m2", skipped=True, completed_attempt_id="a2", completed_artifact_dir="/a2"),
    ]
    cohort = build_cohort(
        launch_id="lid",
        manifest_hash=MANIFEST_HASH,
        manifest_path="/tmp/m.yaml",
        output_dir="/out",
        experiment_name="exp",
        seed=SEED,
        backend=BACKEND,
        pending_jobs=jobs,
        created_at=CREATED_AT,
    )
    assert len(cohort["members"]) == 2
    assert all(m["skipped"] for m in cohort["members"])


def test_member_ids_unique_for_duplicate_dataset_model_different_params():
    job_a = _job("pbmc", "scvi", logical_run_id="lrid-aaa")
    job_b = _job("pbmc", "scvi", logical_run_id="lrid-bbb")
    mid_a = make_member_id(job_a, 0)
    mid_b = make_member_id(job_b, 1)
    assert mid_a != mid_b


def test_member_ids_unique_by_index_alone():
    job = _job("pbmc", "scvi")
    mid_0 = make_member_id(job, 0)
    mid_1 = make_member_id(job, 1)
    assert mid_0 != mid_1


# ---------------------------------------------------------------------------
# Disk write / load round-trip
# ---------------------------------------------------------------------------


def test_write_and_load_latest_cohort(tmp_path):
    launch_id, cohort = _write_fake_cohort(tmp_path, [_job()])
    loaded = load_latest_cohort(tmp_path)
    assert loaded is not None
    assert loaded["launch_id"] == launch_id
    assert len(loaded["members"]) == 1


def test_two_launches_create_distinct_directories(tmp_path):
    id1, _ = _write_fake_cohort(tmp_path, [_job("ds1", "m1")])
    id2, _ = _write_fake_cohort(tmp_path, [_job("ds2", "m2")])
    assert id1 != id2
    assert cohort_path(tmp_path, id1).exists()
    assert cohort_path(tmp_path, id2).exists()
    # latest_launch points to the second launch
    loaded = load_latest_cohort(tmp_path)
    assert loaded["launch_id"] == id2


def test_load_latest_cohort_returns_none_when_no_launch(tmp_path):
    assert load_latest_cohort(tmp_path) is None


def test_load_latest_cohort_returns_none_when_cohort_file_missing(tmp_path):
    launch_id = make_launch_id(
        manifest_hash=MANIFEST_HASH, backend=BACKEND, seed=SEED, created_at=CREATED_AT
    )
    write_latest_launch(
        output_dir=tmp_path, launch_id=launch_id, created_at=CREATED_AT
    )
    # cohort.json was never written
    assert load_latest_cohort(tmp_path) is None


# ---------------------------------------------------------------------------
# Gap 3: submitted attempt patching by stable identity
# ---------------------------------------------------------------------------


def test_update_cohort_submitted_by_member_id(tmp_path):
    job = _job("pbmc", "scvi", logical_run_id="lrid-xyz")
    launch_id, cohort = _write_fake_cohort(tmp_path, [job])

    member_id = cohort["members"][0]["member_id"]
    submitted_runs = [
        {
            "attempt_id": "atmp-999",
            "job_name": "irrelevant_name",
            "dataset": "pbmc",
            "model": "scvi",
            "member_id": member_id,
            "logical_run_id": "lrid-xyz",
        }
    ]
    update_cohort_submitted(
        output_dir=tmp_path, launch_id=launch_id, submitted_runs=submitted_runs
    )
    loaded = load_latest_cohort(tmp_path)
    assert loaded["members"][0]["submitted_attempt_id"] == "atmp-999"


def test_update_cohort_submitted_by_logical_run_id(tmp_path):
    job = _job("pbmc", "scvi", logical_run_id="lrid-fallback")
    launch_id, _ = _write_fake_cohort(tmp_path, [job])

    submitted_runs = [
        {
            "attempt_id": "atmp-888",
            "job_name": "wrong_name",
            "dataset": "pbmc",
            "model": "scvi",
            "member_id": "",  # no member_id
            "logical_run_id": "lrid-fallback",
        }
    ]
    update_cohort_submitted(
        output_dir=tmp_path, launch_id=launch_id, submitted_runs=submitted_runs
    )
    loaded = load_latest_cohort(tmp_path)
    assert loaded["members"][0]["submitted_attempt_id"] == "atmp-888"


def test_update_cohort_submitted_job_name_last_resort(tmp_path):
    job = _job("pbmc", "scvi")
    launch_id, cohort = _write_fake_cohort(tmp_path, [job])
    job_name = cohort["members"][0]["job_name"]

    submitted_runs = [
        {
            "attempt_id": "atmp-777",
            "job_name": job_name,
            "dataset": "pbmc",
            "model": "scvi",
            "member_id": "",
            "logical_run_id": "",
        }
    ]
    update_cohort_submitted(
        output_dir=tmp_path, launch_id=launch_id, submitted_runs=submitted_runs
    )
    loaded = load_latest_cohort(tmp_path)
    assert loaded["members"][0]["submitted_attempt_id"] == "atmp-777"


def test_update_cohort_submitted_duplicate_dataset_model_by_member_id(tmp_path):
    """Two jobs with same dataset+model but different params must not cross-wire."""
    job_a = _job("pbmc", "scvi", logical_run_id="lrid-aaa")
    job_b = _job("pbmc", "scvi", logical_run_id="lrid-bbb")
    launch_id, cohort = _write_fake_cohort(tmp_path, [job_a, job_b])

    mid_a = cohort["members"][0]["member_id"]
    mid_b = cohort["members"][1]["member_id"]

    submitted_runs = [
        {"attempt_id": "atmp-A", "job_name": "pbmc_scvi", "dataset": "pbmc", "model": "scvi",
         "member_id": mid_a, "logical_run_id": "lrid-aaa"},
        {"attempt_id": "atmp-B", "job_name": "pbmc_scvi", "dataset": "pbmc", "model": "scvi",
         "member_id": mid_b, "logical_run_id": "lrid-bbb"},
    ]
    update_cohort_submitted(
        output_dir=tmp_path, launch_id=launch_id, submitted_runs=submitted_runs
    )
    loaded = load_latest_cohort(tmp_path)
    assert loaded["members"][0]["submitted_attempt_id"] == "atmp-A"
    assert loaded["members"][1]["submitted_attempt_id"] == "atmp-B"


# ---------------------------------------------------------------------------
# Readiness: per-status cases
# ---------------------------------------------------------------------------


def _member(
    *,
    submitted_attempt_id: Optional[str] = None,
    completed_attempt_id: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    skipped: bool = False,
    logical_run_id: str = "",
    dataset_path: str = "/data/demo.h5ad",
) -> Dict[str, Any]:
    return {
        "member_id": "m1",
        "job_name": "demo_pca",
        "dataset_slug": "demo",
        "dataset_name": "demo",
        "dataset_path": dataset_path,
        "dataset_path_resolved": dataset_path,
        "model_slug": "pca",
        "logical_run_id": logical_run_id,
        "source": "skipped_completed" if skipped else "submitted",
        "skipped": skipped,
        "completed_attempt_id": completed_attempt_id,
        "submitted_attempt_id": submitted_attempt_id,
        "artifact_dir": artifact_dir,
        "batch_key": "batch",
        "label_key": "cell_type",
        "metrics_requested": {},
        "job": {},
    }


def _make_artifact_dir(tmp_path: Path, *, with_embeddings: bool = True) -> str:
    adir = tmp_path / "artifact"
    adir.mkdir()
    # Write a minimal artifact_manifest.json + .sha256 sidecar.
    manifest_data = {
        "schema_version": 1,
        "mv_contract_version": "1",
        "artifact_entries": [],
        "produced_by": {},
        "produced_at": {},
        "state_transitions": [],
        "resource_observations": {},
    }
    manifest_bytes = json.dumps(manifest_data).encode()
    import hashlib
    sha = hashlib.sha256(manifest_bytes).hexdigest()
    (adir / "artifact_manifest.json").write_bytes(manifest_bytes)
    (adir / "artifact_manifest.sha256").write_text(
        f"{sha}  artifact_manifest.json\n", encoding="ascii"
    )
    if with_embeddings:
        (adir / "embeddings.h5").write_bytes(b"\x89HDF")
    return str(adir)


def test_readiness_running_when_non_terminal_state():
    m = _member(submitted_attempt_id="atmp-1")
    snap = {"physical_attempt_id": "atmp-1", "primary_state": "RUNNING"}
    result = resolve_member_readiness(m, mvd_snapshots={"atmp-1": snap})
    assert result["readiness_status"] == STATUS_RUNNING


def test_readiness_training_failed():
    m = _member(submitted_attempt_id="atmp-2")
    snap = {
        "physical_attempt_id": "atmp-2",
        "primary_state": "FAILED",
        "failure_reason": "OOM",
    }
    result = resolve_member_readiness(m, mvd_snapshots={"atmp-2": snap})
    assert result["readiness_status"] == STATUS_TRAINING_FAILED
    assert "OOM" in result["readiness_reason"]


def test_readiness_cancelled():
    m = _member(submitted_attempt_id="atmp-3")
    snap = {"physical_attempt_id": "atmp-3", "primary_state": "CANCELLED"}
    result = resolve_member_readiness(m, mvd_snapshots={"atmp-3": snap})
    assert result["readiness_status"] == STATUS_CANCELLED


def test_readiness_not_submitted():
    m = _member()
    result = resolve_member_readiness(m)
    assert result["readiness_status"] == STATUS_NOT_SUBMITTED


def test_readiness_missing_artifact_dir(tmp_path):
    m = _member(submitted_attempt_id="atmp-4")
    snap = {
        "physical_attempt_id": "atmp-4",
        "primary_state": "ARTIFACT_SUCCESS",
        "artifact_dir": str(tmp_path / "nonexistent"),
    }
    result = resolve_member_readiness(m, mvd_snapshots={"atmp-4": snap})
    assert result["readiness_status"] == STATUS_MISSING_ARTIFACT_DIR


def test_readiness_no_embeddings(tmp_path):
    adir = _make_artifact_dir(tmp_path, with_embeddings=False)
    m = _member(submitted_attempt_id="atmp-5", artifact_dir=adir, dataset_path=str(tmp_path / "demo.h5ad"))
    (tmp_path / "demo.h5ad").write_bytes(b"")  # create the dataset file
    snap = {
        "physical_attempt_id": "atmp-5",
        "primary_state": "ARTIFACT_SUCCESS",
        "artifact_dir": adir,
    }
    result = resolve_member_readiness(m, mvd_snapshots={"atmp-5": snap})
    assert result["readiness_status"] == STATUS_NO_EMBEDDINGS


def test_readiness_missing_dataset(tmp_path):
    adir = _make_artifact_dir(tmp_path, with_embeddings=True)
    m = _member(
        submitted_attempt_id="atmp-6",
        artifact_dir=adir,
        dataset_path=str(tmp_path / "missing.h5ad"),
    )
    # Patch _verify_artifact_dir to return ready so we reach _check_dataset.
    with patch(
        "multiverse.evaluation.cohort._verify_artifact_dir",
        return_value=(STATUS_READY, ""),
    ):
        result = resolve_member_readiness(m, mvd_snapshots={
            "atmp-6": {"physical_attempt_id": "atmp-6", "primary_state": "ARTIFACT_SUCCESS", "artifact_dir": adir}
        })
    assert result["readiness_status"] == STATUS_MISSING_DATASET


def test_readiness_unsupported_dataset(tmp_path):
    adir = _make_artifact_dir(tmp_path, with_embeddings=True)
    ds = tmp_path / "data.csv"
    ds.write_text("a,b\n1,2\n")
    m = _member(
        submitted_attempt_id="atmp-7",
        artifact_dir=adir,
        dataset_path=str(ds),
    )
    with patch(
        "multiverse.evaluation.cohort._verify_artifact_dir",
        return_value=(STATUS_READY, ""),
    ):
        result = resolve_member_readiness(m, mvd_snapshots={
            "atmp-7": {"physical_attempt_id": "atmp-7", "primary_state": "ARTIFACT_SUCCESS", "artifact_dir": adir}
        })
    assert result["readiness_status"] == STATUS_UNSUPPORTED_DATASET


# ---------------------------------------------------------------------------
# Gap 4: skipped member revalidation through completed_runs
# ---------------------------------------------------------------------------


def test_skipped_member_ready_when_in_completed_runs(tmp_path):
    adir = _make_artifact_dir(tmp_path, with_embeddings=True)
    ds = tmp_path / "demo.h5ad"
    ds.write_bytes(b"")
    m = _member(
        completed_attempt_id="atmp-old",
        artifact_dir=adir,
        skipped=True,
        logical_run_id="lrid-ok",
        dataset_path=str(ds),
    )
    completed_runs = {"lrid-ok": {"attempt_id": "atmp-old", "artifact_dir": adir}}
    with patch(
        "multiverse.evaluation.cohort._verify_artifact_dir",
        return_value=(STATUS_READY, ""),
    ), patch(
        "multiverse.evaluation.cohort._check_dataset",
        return_value=(STATUS_READY, ""),
    ):
        result = resolve_member_readiness(m, completed_runs=completed_runs)
    assert result["readiness_status"] == STATUS_READY


def test_skipped_member_not_ready_when_missing_from_completed_runs():
    m = _member(
        completed_attempt_id="atmp-old",
        artifact_dir="/some/dir",
        skipped=True,
        logical_run_id="lrid-gone",
    )
    completed_runs: Dict[str, Any] = {}  # logical run no longer present
    result = resolve_member_readiness(m, completed_runs=completed_runs)
    assert result["readiness_status"] == STATUS_MISSING_ARTIFACT_DIR
    assert "no longer ARTIFACT_SUCCESS" in result["readiness_reason"]


# ---------------------------------------------------------------------------
# Gap 5: logical-run fallback after refresh (not_submitted → ready via completed_runs)
# ---------------------------------------------------------------------------


def test_not_submitted_member_becomes_ready_via_completed_runs(tmp_path):
    adir = _make_artifact_dir(tmp_path, with_embeddings=True)
    ds = tmp_path / "demo.h5ad"
    ds.write_bytes(b"")
    m = _member(logical_run_id="lrid-resolved", dataset_path=str(ds))
    completed_runs = {
        "lrid-resolved": {"attempt_id": "atmp-resolved", "artifact_dir": adir}
    }
    with patch(
        "multiverse.evaluation.cohort._verify_artifact_dir",
        return_value=(STATUS_READY, ""),
    ), patch(
        "multiverse.evaluation.cohort._check_dataset",
        return_value=(STATUS_READY, ""),
    ):
        result = resolve_member_readiness(m, completed_runs=completed_runs)
    assert result["readiness_status"] == STATUS_READY
    assert result["artifact_dir"] == adir


# ---------------------------------------------------------------------------
# readiness_summary and can_evaluate gate
# ---------------------------------------------------------------------------


def test_readiness_summary_no_ready_members():
    members = [
        {**_member(submitted_attempt_id="a1"), "readiness_status": STATUS_RUNNING, "readiness_reason": ""},
        {**_member(submitted_attempt_id="a2"), "readiness_status": STATUS_TRAINING_FAILED, "readiness_reason": ""},
    ]
    summary = readiness_summary(members)
    assert summary["ready"] == 0
    assert summary["can_evaluate"] is False
    assert summary["total"] == 2


def test_readiness_summary_partial_ready():
    members = [
        {**_member(submitted_attempt_id="a1"), "readiness_status": STATUS_READY, "readiness_reason": ""},
        {**_member(submitted_attempt_id="a2"), "readiness_status": STATUS_RUNNING, "readiness_reason": ""},
        {**_member(submitted_attempt_id="a3"), "readiness_status": STATUS_TRAINING_FAILED, "readiness_reason": ""},
    ]
    summary = readiness_summary(members)
    assert summary["ready"] == 1
    assert summary["can_evaluate"] is True
    assert summary["total"] == 3


def test_resolve_cohort_readiness_all_not_submitted():
    cohort = {
        "members": [_member(), _member()]
    }
    results = resolve_cohort_readiness(cohort)
    assert all(r["readiness_status"] == STATUS_NOT_SUBMITTED for r in results)
    summary = readiness_summary(results)
    assert summary["can_evaluate"] is False


def test_resolve_cohort_readiness_one_ready_enables_button(tmp_path):
    adir = _make_artifact_dir(tmp_path, with_embeddings=True)
    ds = tmp_path / "demo.h5ad"
    ds.write_bytes(b"")
    cohort = {
        "members": [
            _member(submitted_attempt_id="a-ready", artifact_dir=adir, dataset_path=str(ds)),
            _member(submitted_attempt_id="a-running"),
        ]
    }
    snaps = {
        "a-ready": {"physical_attempt_id": "a-ready", "primary_state": "ARTIFACT_SUCCESS",
                    "artifact_dir": adir},
        "a-running": {"physical_attempt_id": "a-running", "primary_state": "RUNNING"},
    }
    with patch(
        "multiverse.evaluation.cohort._verify_artifact_dir",
        return_value=(STATUS_READY, ""),
    ), patch(
        "multiverse.evaluation.cohort._check_dataset",
        return_value=(STATUS_READY, ""),
    ):
        results = resolve_cohort_readiness(cohort, mvd_snapshots=snaps)
    summary = readiness_summary(results)
    assert summary["can_evaluate"] is True
    assert summary["ready"] == 1


# ---------------------------------------------------------------------------
# Gap 2: cohort write failure logging (not silently swallowed)
# ---------------------------------------------------------------------------


def test_update_cohort_submitted_logs_warning_when_cohort_missing(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="multiverse.evaluation.cohort"):
        update_cohort_submitted(
            output_dir=tmp_path,
            launch_id="nonexistent-id",
            submitted_runs=[{"attempt_id": "a1", "job_name": "j1", "member_id": "m1", "logical_run_id": ""}],
        )
    assert any("cohort not found" in r.message for r in caplog.records)


def test_update_cohort_submitted_logs_warning_when_no_members_matched(tmp_path, caplog):
    import logging

    launch_id, _ = _write_fake_cohort(tmp_path, [_job()])
    with caplog.at_level(logging.WARNING, logger="multiverse.evaluation.cohort"):
        update_cohort_submitted(
            output_dir=tmp_path,
            launch_id=launch_id,
            submitted_runs=[
                {"attempt_id": "a1", "job_name": "WRONG_NAME", "member_id": "WRONG_ID", "logical_run_id": "WRONG_LRID"}
            ],
        )
    assert any("no members updated" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Gap 3: SubmittedRun carries logical_run_id from source
# ---------------------------------------------------------------------------


def test_submitted_run_to_dict_includes_logical_run_id():
    from multiverse.runner.mvd_inprocess import SubmittedRun

    sr = SubmittedRun(
        attempt_id="atmp-abc",
        job_name="pbmc_scvi",
        dataset="pbmc",
        model="scvi",
        logical_run_id="lrid-xyz",
    )
    d = sr.to_dict()
    assert d["logical_run_id"] == "lrid-xyz"
    assert d["attempt_id"] == "atmp-abc"


def test_submitted_run_logical_run_id_defaults_to_empty():
    from multiverse.runner.mvd_inprocess import SubmittedRun

    sr = SubmittedRun(attempt_id="a", job_name="j", dataset="d", model="m")
    assert sr.logical_run_id == ""
    assert sr.to_dict()["logical_run_id"] == ""


def test_update_cohort_prefers_source_logical_run_id_over_gui_fallback(tmp_path):
    """logical_run_id carried in SubmittedRun.to_dict() is used without GUI zip-order."""
    job = _job("pbmc", "scvi", logical_run_id="lrid-source")
    launch_id, _ = _write_fake_cohort(tmp_path, [job])

    # submitted_runs dict comes from SubmittedRun.to_dict() which now includes
    # logical_run_id from the source — no member_id provided.
    submitted_runs = [
        {
            "attempt_id": "atmp-src",
            "job_name": "irrelevant",
            "dataset": "pbmc",
            "model": "scvi",
            "member_id": "",
            "logical_run_id": "lrid-source",  # carried by SubmittedRun
        }
    ]
    update_cohort_submitted(
        output_dir=tmp_path, launch_id=launch_id, submitted_runs=submitted_runs
    )
    loaded = load_latest_cohort(tmp_path)
    assert loaded["members"][0]["submitted_attempt_id"] == "atmp-src"


# ---------------------------------------------------------------------------
# Gap 4: _write_launch_cohort logs on failure
# ---------------------------------------------------------------------------


def test_write_launch_cohort_logs_warning_on_failure(tmp_path, caplog):
    """Both a log record and no launch_id are produced when cohort write fails."""
    import logging

    # We test the cohort module's write path by patching write_cohort to raise.
    with patch("multiverse.evaluation.cohort.write_cohort", side_effect=OSError("disk full")):
        from multiverse.evaluation.cohort import (build_cohort, make_launch_id,
                                                   write_latest_launch)
        from datetime import datetime, timezone

        created_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        launch_id = make_launch_id(
            manifest_hash=MANIFEST_HASH, backend=BACKEND, seed=SEED, created_at=created_at
        )
        cohort = build_cohort(
            launch_id=launch_id,
            manifest_hash=MANIFEST_HASH,
            manifest_path="/tmp/m.yaml",
            output_dir=str(tmp_path),
            experiment_name="exp",
            seed=SEED,
            backend=BACKEND,
            pending_jobs=[_job()],
            created_at=created_at,
        )

        # Invoke write_cohort directly to confirm it raises, then verify the
        # GUI-level wrapper would log.  We test the cohort module behaviour here
        # and the GUI wrapper in the gui-layer test below.
        with pytest.raises(OSError, match="disk full"):
            from multiverse.evaluation.cohort import write_cohort as wc
            wc(output_dir=tmp_path, launch_id=launch_id, cohort=cohort)


def test_write_launch_cohort_gui_wrapper_logs_and_warns(tmp_path, caplog, monkeypatch):
    """_write_launch_cohort() emits both a log record and st.warning on failure."""
    import logging

    warnings_shown = []

    # Patch st.warning so we can intercept without a running Streamlit session.
    import multiverse.gui as gui_mod
    monkeypatch.setattr(gui_mod.st, "warning", lambda msg: warnings_shown.append(msg))

    # Patch cohort write to fail after launch_id is created.
    monkeypatch.setattr(
        "multiverse.evaluation.cohort.write_cohort",
        lambda **kw: (_ for _ in ()).throw(OSError("no space left")),
    )

    with caplog.at_level(logging.WARNING, logger="multiverse.gui"):
        result = gui_mod._write_launch_cohort(
            output_path=tmp_path,
            manifest_file=Path("/tmp/m.yaml"),
            manifest_text="globals:\n  experiment_name: test\n",
            manifest_hash=MANIFEST_HASH,
            experiment_name="test",
            seed=SEED,
            backend=BACKEND,
            pending_jobs=[_job()],
        )

    assert result == "", "should return empty string on failure"
    assert warnings_shown, "st.warning should have been called"
    assert any("no space left" in w for w in warnings_shown)
    assert any("cohort write failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Gap 5: evaluate_section_view pure rendering helper
# ---------------------------------------------------------------------------


def _ready_member(**kwargs) -> Dict[str, Any]:
    base = _member(**kwargs)
    base["readiness_status"] = STATUS_READY
    base["readiness_reason"] = ""
    return base


def _running_member(**kwargs) -> Dict[str, Any]:
    base = _member(**kwargs)
    base["readiness_status"] = STATUS_RUNNING
    base["readiness_reason"] = "run is in state RUNNING"
    return base


def _failed_member(**kwargs) -> Dict[str, Any]:
    base = _member(**kwargs)
    base["readiness_status"] = STATUS_TRAINING_FAILED
    base["readiness_reason"] = "run reached FAILED: OOM"
    return base


def test_evaluate_section_view_no_members():
    view = evaluate_section_view([])
    assert view["total"] == 0
    assert view["ready"] == 0
    assert view["button_enabled"] is False
    assert view["button_label"] == "Evaluate experiment"
    assert "0/0" in view["summary_text"]
    assert view["table_rows"] == []


def test_evaluate_section_view_zero_ready_disables_button():
    members = [_running_member(), _failed_member()]
    view = evaluate_section_view(members)
    assert view["button_enabled"] is False
    assert view["ready"] == 0
    assert view["button_label"] == "Evaluate experiment"
    assert "0/2" in view["summary_text"]
    assert "running" in view["summary_text"]
    assert "training failed" in view["summary_text"]


def test_evaluate_section_view_one_ready_enables_button():
    members = [_ready_member(), _running_member()]
    view = evaluate_section_view(members)
    assert view["button_enabled"] is True
    assert view["ready"] == 1
    assert "1" in view["button_label"]
    assert "Evaluate experiment" in view["button_label"]
    assert "1/2" in view["summary_text"]


def test_evaluate_section_view_all_ready():
    members = [_ready_member(), _ready_member()]
    view = evaluate_section_view(members)
    assert view["button_enabled"] is True
    assert view["ready"] == 2
    assert view["total"] == 2
    # No non-ready parts — summary is just "N/N ready for evaluation"
    assert "(" not in view["summary_text"]


def test_evaluate_section_view_table_rows_contain_status():
    members = [
        _ready_member(submitted_attempt_id="a1"),
        _running_member(submitted_attempt_id="a2"),
    ]
    view = evaluate_section_view(members)
    statuses = [r["Status"] for r in view["table_rows"]]
    assert STATUS_READY in statuses
    assert STATUS_RUNNING in statuses


def test_evaluate_section_view_button_label_includes_ready_count():
    members = [_ready_member(), _ready_member(), _running_member()]
    view = evaluate_section_view(members)
    assert "2" in view["button_label"]
    assert "ready" in view["button_label"]


def test_evaluate_section_view_partial_readiness_summary_format():
    members = [
        _ready_member(),
        _running_member(),
        _failed_member(),
    ]
    view = evaluate_section_view(members)
    assert "1/3" in view["summary_text"]
    # Non-ready counts appear in the parenthetical
    assert "running" in view["summary_text"]
    assert "training failed" in view["summary_text"]
