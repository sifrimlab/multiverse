"""Per-member evaluation results and the launch-level evaluation report.

This module is the source of truth for evaluation *outcomes* (as opposed to
:mod:`multiverse.evaluation.cohort`, which owns pre-evaluation *readiness*).

Two artifacts are written under the launch directory
(``<output-dir>/.multiverse/launches/<launch_id>/``)::

    evaluations/<member_id>.json   # one structured result per cohort member
    evaluation_report.json         # derived launch-level aggregation

Like :mod:`cohort`, this module imports no heavy scientific dependencies, so it
is safe to import in a thin host environment (GUI, aggregation, tests) as well
as inside the evaluation container that writes the per-member files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cohort import _atomic_write_json, launch_dir

logger = logging.getLogger(__name__)

EVALUATION_REPORT_SCHEMA_VERSION = 1
MEMBER_RESULT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Evaluation status vocabulary (post-evaluation, per member).
# Distinct from the readiness vocabulary in cohort.py (pre-evaluation).
# ---------------------------------------------------------------------------

EVAL_STATUS_PENDING = "pending"
EVAL_STATUS_RUNNING = "running"
EVAL_STATUS_DONE = "done"
EVAL_STATUS_TRAINING_FAILED = "training_failed"
EVAL_STATUS_NOT_READY = "not_ready"
EVAL_STATUS_NO_EMBEDDINGS = "no_embeddings"
EVAL_STATUS_MISSING_DATASET = "missing_dataset"
EVAL_STATUS_BAD_MANIFEST = "bad_manifest"
EVAL_STATUS_OBS_MISMATCH = "obs_mismatch"
EVAL_STATUS_UNSUPPORTED_DATASET = "unsupported_dataset"
EVAL_STATUS_EVALUATION_FAILED = "evaluation_failed"

EVAL_STATUSES = frozenset(
    {
        EVAL_STATUS_PENDING,
        EVAL_STATUS_RUNNING,
        EVAL_STATUS_DONE,
        EVAL_STATUS_TRAINING_FAILED,
        EVAL_STATUS_NOT_READY,
        EVAL_STATUS_NO_EMBEDDINGS,
        EVAL_STATUS_MISSING_DATASET,
        EVAL_STATUS_BAD_MANIFEST,
        EVAL_STATUS_OBS_MISMATCH,
        EVAL_STATUS_UNSUPPORTED_DATASET,
        EVAL_STATUS_EVALUATION_FAILED,
    }
)

# Readiness-status → evaluation-status projection. When a member never reaches
# the container (filtered out by readiness), the aggregator still represents it
# in the report by projecting its readiness status onto the evaluation
# vocabulary. Readiness ``ready`` projects to ``pending`` (evaluable, not yet
# evaluated); unmapped readiness statuses fall back to ``not_ready``.
_READINESS_TO_EVAL = {
    "ready": EVAL_STATUS_PENDING,
    "running": EVAL_STATUS_NOT_READY,
    "training_failed": EVAL_STATUS_TRAINING_FAILED,
    "cancelled": EVAL_STATUS_TRAINING_FAILED,
    "not_submitted": EVAL_STATUS_NOT_READY,
    "missing_artifact_dir": EVAL_STATUS_NOT_READY,
    "bad_artifact_manifest": EVAL_STATUS_BAD_MANIFEST,
    "no_embeddings": EVAL_STATUS_NO_EMBEDDINGS,
    "missing_dataset": EVAL_STATUS_MISSING_DATASET,
    "unsupported_dataset": EVAL_STATUS_UNSUPPORTED_DATASET,
}


def readiness_to_eval_status(readiness_status: Optional[str]) -> str:
    """Project a cohort readiness status onto the evaluation vocabulary."""
    return _READINESS_TO_EVAL.get(readiness_status or "", EVAL_STATUS_NOT_READY)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def evaluations_dir(output_dir: Path, launch_id: str) -> Path:
    return launch_dir(output_dir, launch_id) / "evaluations"


def member_result_path(output_dir: Path, launch_id: str, member_id: str) -> Path:
    return evaluations_dir(output_dir, launch_id) / f"{member_id}.json"


def report_path(output_dir: Path, launch_id: str) -> Path:
    return launch_dir(output_dir, launch_id) / "evaluation_report.json"


# ---------------------------------------------------------------------------
# Per-member result
# ---------------------------------------------------------------------------


@dataclass
class MemberResult:
    """A structured evaluation outcome for one cohort member.

    Mirrors the spec's ``evaluations/<member_id>.json`` schema. ``metrics`` holds
    the scIB ``evaluation`` block (or model-native metrics); ``error`` holds
    structured failure details (``type``, ``message``, optional ``traceback``).
    """

    member_id: str
    status: str
    reason: str = ""
    artifact_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    schema_version: int = MEMBER_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemberResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def write_member_result(
    *, output_dir: Path, launch_id: str, result: MemberResult
) -> Path:
    """Atomically write ``evaluations/<member_id>.json`` for one member."""
    path = member_result_path(output_dir, launch_id, result.member_id)
    _atomic_write_json(path, result.to_dict())
    return path


def load_member_results(output_dir: Path, launch_id: str) -> List[MemberResult]:
    """Load all per-member result files for a launch (missing dir → empty)."""
    d = evaluations_dir(output_dir, launch_id)
    if not d.is_dir():
        return []
    out: List[MemberResult] = []
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                out.append(MemberResult.from_dict(json.load(fh)))
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort
            logger.warning("load_member_results: could not read %s: %s", p, exc)
    return out


# ---------------------------------------------------------------------------
# Launch-level report (derived)
# ---------------------------------------------------------------------------


def build_evaluation_report(
    *,
    cohort: Dict[str, Any],
    members_with_status: List[Dict[str, Any]],
    member_results: List[MemberResult],
) -> Dict[str, Any]:
    """Aggregate per-member results into the launch-level report.

    The report is fully derived from (a) the cohort, (b) readiness-resolved
    members, and (c) the per-member result files. Members that were evaluated
    use their result's ``status``/``metrics``; members that were never evaluated
    are projected from their readiness status (ready → ``pending``).
    """
    results_by_id = {r.member_id: r for r in member_results}
    readiness_by_id = {
        m.get("member_id"): m for m in members_with_status if m.get("member_id")
    }

    readiness_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    table: List[Dict[str, Any]] = []

    for member in cohort.get("members", []):
        mid = member.get("member_id")
        readiness = readiness_by_id.get(mid, {})
        readiness_status = readiness.get("readiness_status") or "unknown"
        readiness_counts[readiness_status] = (
            readiness_counts.get(readiness_status, 0) + 1
        )

        result = results_by_id.get(mid)
        if result is not None:
            eval_status = result.status
            reason = result.reason
            metrics = result.metrics or {}
            artifact_dir = result.artifact_dir or readiness.get("artifact_dir") or member.get("artifact_dir")
        else:
            eval_status = readiness_to_eval_status(readiness_status)
            reason = readiness.get("readiness_reason") or ""
            metrics = {}
            artifact_dir = readiness.get("artifact_dir") or member.get("artifact_dir")

        status_counts[eval_status] = status_counts.get(eval_status, 0) + 1
        table.append(
            {
                "member_id": mid,
                "dataset": member.get("dataset_slug") or member.get("dataset_name") or "",
                "model": member.get("model_slug") or "",
                "source": member.get("source") or "",
                "readiness_status": readiness_status,
                "eval_status": eval_status,
                "reason": reason,
                "artifact_dir": artifact_dir or "",
                "metrics": metrics,
            }
        )

    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "launch_id": cohort.get("launch_id"),
        "manifest_hash": cohort.get("manifest_hash"),
        "backend": cohort.get("backend"),
        "seed": cohort.get("seed"),
        "experiment_name": cohort.get("experiment_name"),
        "created_at": cohort.get("created_at"),
        "total": len(cohort.get("members", [])),
        "readiness_counts": readiness_counts,
        "status_counts": status_counts,
        "table": table,
    }


def report_to_table_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a report into display rows (one per member, metrics expanded).

    Pure (no Streamlit) so the GUI's comparison table is unit-testable. Each row
    has Dataset/Model/Status/Reason, one column per scIB metric from the
    member's ``evaluation`` block, and the artifact dir for drill-down.
    """
    rows: List[Dict[str, Any]] = []
    for entry in report.get("table", []):
        row: Dict[str, Any] = {
            "Dataset": entry.get("dataset", ""),
            "Model": entry.get("model", ""),
            "Status": entry.get("eval_status", ""),
            "Reason": entry.get("reason", ""),
        }
        evblock = (entry.get("metrics") or {}).get("evaluation") or {}
        for metric_name, val in evblock.items():
            row[metric_name] = val
        row["Artifact dir"] = entry.get("artifact_dir") or ""
        rows.append(row)
    return rows


def write_evaluation_report(
    *, output_dir: Path, launch_id: str, report: Dict[str, Any]
) -> Path:
    """Atomically write ``evaluation_report.json`` under the launch dir."""
    path = report_path(output_dir, launch_id)
    _atomic_write_json(path, report)
    return path


def load_evaluation_report(output_dir: Path, launch_id: str) -> Optional[Dict[str, Any]]:
    """Load ``evaluation_report.json`` for a launch, or None if absent/unreadable."""
    path = report_path(output_dir, launch_id)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_evaluation_report: could not read %s: %s", path, exc)
        return None
