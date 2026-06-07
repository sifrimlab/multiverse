"""Tests for multiverse/evaluation/result.py — per-member result files and the
derived launch-level evaluation report. Host-safe: no heavy scientific deps.
"""

from __future__ import annotations

import json
from pathlib import Path

from multiverse.evaluation import result as r
from multiverse.evaluation.cohort import launch_dir
from multiverse.evaluation.result import (MemberResult,
                                          build_evaluation_report,
                                          load_evaluation_report,
                                          load_member_results,
                                          readiness_to_eval_status,
                                          write_evaluation_report,
                                          write_member_result)

LID = "abc_docker_seed42_20260607T000000_deadbe"


def _member(mid, dataset="ds1", model="pca", source="submitted"):
    return {
        "member_id": mid,
        "dataset_slug": dataset,
        "dataset_name": dataset,
        "model_slug": model,
        "source": source,
        "artifact_dir": f"/art/{mid}",
    }


def _cohort(members):
    return {
        "launch_id": LID,
        "manifest_hash": "abc12345",
        "backend": "docker",
        "seed": 42,
        "experiment_name": "exp",
        "created_at": "2026-06-07T00:00:00Z",
        "members": members,
    }


# --- per-member result round trip ------------------------------------------


def test_member_result_round_trip(tmp_path):
    res = MemberResult(
        member_id="m1",
        status=r.EVAL_STATUS_DONE,
        reason="",
        artifact_dir="/art/m1",
        dataset_path="/d/ds1.h5mu",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:05Z",
        duration_seconds=5.0,
        metrics={"evaluation": {"ARI": 0.8}},
    )
    path = write_member_result(output_dir=tmp_path, launch_id=LID, result=res)
    assert path == launch_dir(tmp_path, LID) / "evaluations" / "m1.json"
    with open(path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["status"] == "done"
    assert on_disk["metrics"]["evaluation"]["ARI"] == 0.8

    loaded = load_member_results(tmp_path, LID)
    assert len(loaded) == 1
    assert loaded[0].member_id == "m1"
    assert loaded[0].status == r.EVAL_STATUS_DONE


def test_load_member_results_missing_dir(tmp_path):
    assert load_member_results(tmp_path, LID) == []


def test_from_dict_ignores_unknown_fields():
    res = MemberResult.from_dict(
        {"member_id": "m", "status": "done", "bogus": 1, "extra": "x"}
    )
    assert res.member_id == "m"
    assert res.status == "done"


# --- readiness projection ---------------------------------------------------


def test_readiness_to_eval_status_projection():
    assert readiness_to_eval_status("ready") == r.EVAL_STATUS_PENDING
    assert readiness_to_eval_status("training_failed") == r.EVAL_STATUS_TRAINING_FAILED
    assert readiness_to_eval_status("no_embeddings") == r.EVAL_STATUS_NO_EMBEDDINGS
    assert readiness_to_eval_status("something_new") == r.EVAL_STATUS_NOT_READY
    assert readiness_to_eval_status(None) == r.EVAL_STATUS_NOT_READY


# --- report aggregation -----------------------------------------------------


def test_build_report_mixes_results_and_pending():
    """Evaluated members use their result status; an evaluable-but-unevaluated
    member is reported as ``pending``; a failed-training member is projected."""
    members = [_member("m1"), _member("m2"), _member("m3")]
    cohort = _cohort(members)
    members_with_status = [
        {**members[0], "readiness_status": "ready", "readiness_reason": ""},
        {**members[1], "readiness_status": "ready", "readiness_reason": ""},
        {**members[2], "readiness_status": "training_failed", "readiness_reason": "boom"},
    ]
    # only m1 was evaluated
    results = [
        MemberResult(member_id="m1", status=r.EVAL_STATUS_DONE, metrics={"evaluation": {"ARI": 0.9}})
    ]
    report = build_evaluation_report(
        cohort=cohort, members_with_status=members_with_status, member_results=results
    )
    assert report["total"] == 3
    assert report["status_counts"]["done"] == 1
    assert report["status_counts"]["pending"] == 1  # m2: ready, no result
    assert report["status_counts"]["training_failed"] == 1  # m3 projected
    assert report["readiness_counts"]["ready"] == 2
    assert report["readiness_counts"]["training_failed"] == 1

    rows = {row["member_id"]: row for row in report["table"]}
    assert rows["m1"]["eval_status"] == "done"
    assert rows["m1"]["metrics"]["evaluation"]["ARI"] == 0.9
    assert rows["m2"]["eval_status"] == "pending"
    assert rows["m3"]["eval_status"] == "training_failed"
    assert rows["m3"]["reason"] == "boom"


def test_report_round_trip(tmp_path):
    cohort = _cohort([_member("m1")])
    report = build_evaluation_report(
        cohort=cohort,
        members_with_status=[{**cohort["members"][0], "readiness_status": "ready"}],
        member_results=[MemberResult(member_id="m1", status=r.EVAL_STATUS_DONE)],
    )
    write_evaluation_report(output_dir=tmp_path, launch_id=LID, report=report)
    loaded = load_evaluation_report(tmp_path, LID)
    assert loaded is not None
    assert loaded["launch_id"] == LID
    assert loaded["status_counts"]["done"] == 1


def test_load_report_absent(tmp_path):
    assert load_evaluation_report(tmp_path, LID) is None


def test_report_to_table_rows_expands_metrics():
    from multiverse.evaluation.result import report_to_table_rows

    report = {
        "table": [
            {
                "member_id": "m1",
                "dataset": "ds1",
                "model": "pca",
                "eval_status": "done",
                "reason": "",
                "artifact_dir": "/art/m1",
                "metrics": {"evaluation": {"ARI": 0.8, "NMI": 0.9}},
            },
            {
                "member_id": "m2",
                "dataset": "ds1",
                "model": "mofa",
                "eval_status": "pending",
                "reason": "not evaluated yet",
                "artifact_dir": "",
                "metrics": {},
            },
        ]
    }
    rows = report_to_table_rows(report)
    assert rows[0]["Dataset"] == "ds1"
    assert rows[0]["Status"] == "done"
    assert rows[0]["ARI"] == 0.8
    assert rows[0]["NMI"] == 0.9
    assert rows[0]["Artifact dir"] == "/art/m1"
    # Member with no metrics still produces a row (status/reason only).
    assert rows[1]["Status"] == "pending"
    assert "ARI" not in rows[1]
