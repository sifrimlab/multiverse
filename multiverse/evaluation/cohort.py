"""Launch cohort persistence and readiness resolution.

On-disk layout under the selected output directory::

    <output-dir>/.multiverse/
      latest_launch.json
      launches/
        <launch_id>/
          manifest.yaml   (copy of manifest text)
          cohort.json

No Streamlit imports — this module is safe to unit-test directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

COHORT_SCHEMA_VERSION = 1
LATEST_LAUNCH_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def multiverse_root(output_dir: Path) -> Path:
    """Return the .multiverse directory under output_dir (not created)."""
    return Path(output_dir) / ".multiverse"


def launches_dir(output_dir: Path) -> Path:
    return multiverse_root(output_dir) / "launches"


def launch_dir(output_dir: Path, launch_id: str) -> Path:
    return launches_dir(output_dir) / launch_id


def cohort_path(output_dir: Path, launch_id: str) -> Path:
    return launch_dir(output_dir, launch_id) / "cohort.json"


def manifest_copy_path(output_dir: Path, launch_id: str) -> Path:
    return launch_dir(output_dir, launch_id) / "manifest.yaml"


def latest_launch_path(output_dir: Path) -> Path:
    return multiverse_root(output_dir) / "latest_launch.json"


# ---------------------------------------------------------------------------
# Launch ID  (Gap 1 — inspectable, context-rich format)
# ---------------------------------------------------------------------------


def make_launch_id(
    *,
    manifest_hash: str,
    backend: str,
    seed: Optional[int],
    created_at: str,
) -> str:
    """Return an inspectable, unique launch ID.

    Format: ``<manifest_prefix>_<backend>_seed<seed>_<yyyymmddThhmmss>_<nonce>``

    Two calls with identical arguments still produce distinct IDs because of the
    random nonce, so repeated launches of the same manifest never collide.
    """
    # Compact the ISO-8601 timestamp: "2026-06-03T12:00:00Z" → "20260603T120000"
    ts = (
        created_at.replace("-", "").replace(":", "").replace("Z", "").split(".")[0]
    )
    seed_str = str(seed) if seed is not None else "none"
    prefix = (manifest_hash or "x")[:8]
    nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:6]
    return f"{prefix}_{backend}_seed{seed_str}_{ts}_{nonce}"


# ---------------------------------------------------------------------------
# Member ID
# ---------------------------------------------------------------------------


def make_member_id(job: Dict[str, Any], index: int) -> str:
    """Deterministic, collision-resistant member ID.

    Combines dataset slug, model slug, logical run ID, and job index so that
    duplicate dataset/model pairs that differ only by parameters get distinct IDs.
    """
    dataset_slug = str(job.get("dataset_slug") or job.get("dataset_name") or "ds")
    model_slug = str(job.get("model_slug") or job.get("model_name") or "mdl")
    logical_run_id = str(job.get("_logical_run_id") or "")
    raw = f"{dataset_slug}|{model_slug}|{logical_run_id}|{index}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{dataset_slug[:12]}-{model_slug[:12]}-{digest}"


# ---------------------------------------------------------------------------
# Dataset path resolution  (Gap 6)
# ---------------------------------------------------------------------------


def _resolve_dataset_path(dataset_path: str) -> Tuple[str, str]:
    """Return (original, resolved_absolute) for a dataset path.

    If the path is already absolute and exists, resolved == original.
    If it is relative, we attempt to resolve it against the repository root
    (two levels above this file).  The returned resolved path is recorded in
    the cohort so readiness checks work after directory changes.
    """
    if not dataset_path:
        return "", ""
    original = dataset_path
    p = Path(dataset_path)
    if p.is_absolute():
        return original, str(p)
    # Resolve against repo root (multiverse/evaluation/cohort.py → ../../..)
    repo_root = Path(__file__).resolve().parents[2]
    resolved = repo_root / p
    return original, str(resolved)


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------


def _job_source(job: Dict[str, Any]) -> str:
    if job.get("_skipped") and job.get("_completed_attempt_id"):
        return "skipped_completed"
    if job.get("_skipped"):
        return "planned_only"
    return "submitted"


def build_cohort(
    *,
    launch_id: str,
    manifest_hash: str,
    manifest_path: str,
    output_dir: str,
    experiment_name: str,
    seed: Optional[int],
    backend: str,
    pending_jobs: List[Dict[str, Any]],
    created_at: str,
) -> Dict[str, Any]:
    """Build the cohort dict from resume-decorated pending jobs.

    All jobs (runnable and skipped) are included so the cohort faithfully
    represents the full launch scope.  Returns the cohort dict AND records the
    member_id for each job at its plan index so callers can map submissions back.
    """
    members = []
    for i, job in enumerate(pending_jobs):
        member_id = make_member_id(job, i)
        source = _job_source(job)
        dp_original, dp_resolved = _resolve_dataset_path(str(job.get("dataset_path") or ""))
        member: Dict[str, Any] = {
            "member_id": member_id,
            "job_name": (
                job.get("name")
                or (
                    f"{job.get('dataset_slug') or job.get('dataset_name') or '?'}_"
                    f"{job.get('model_slug') or job.get('model_name') or '?'}"
                )
            ),
            "dataset_slug": str(job.get("dataset_slug") or job.get("dataset_name") or ""),
            "dataset_name": str(job.get("dataset_name") or job.get("dataset_slug") or ""),
            "dataset_path": dp_original,
            "dataset_path_resolved": dp_resolved,
            "model_slug": str(job.get("model_slug") or job.get("model_name") or ""),
            "logical_run_id": str(job.get("_logical_run_id") or ""),
            "source": source,
            "skipped": bool(job.get("_skipped")),
            "completed_attempt_id": job.get("_completed_attempt_id") or None,
            "submitted_attempt_id": None,
            "artifact_dir": job.get("_completed_artifact_dir") or None,
            "batch_key": str(job.get("batch_key") or "batch"),
            "label_key": str(job.get("cell_type_key") or "cell_type"),
            "metrics_requested": dict(job.get("metrics") or {}),
            "job": {k: v for k, v in job.items() if not k.startswith("_")},
        }
        members.append(member)

    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "launch_id": launch_id,
        "manifest_hash": manifest_hash,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
        "experiment_name": experiment_name,
        "seed": seed,
        "backend": backend,
        "created_at": created_at,
        "members": members,
    }


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically: temp-file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_cohort(
    *,
    output_dir: Path,
    launch_id: str,
    cohort: Dict[str, Any],
    manifest_text: Optional[str] = None,
) -> Path:
    """Write cohort.json (and optionally manifest.yaml) under .multiverse/launches/<launch_id>/."""
    cpath = cohort_path(output_dir, launch_id)
    _atomic_write_json(cpath, cohort)
    if manifest_text is not None:
        mpath = manifest_copy_path(output_dir, launch_id)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=mpath.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(manifest_text)
            os.replace(tmp, mpath)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return cpath


def update_cohort_artifact_dirs(
    *,
    output_dir: Path,
    launch_id: str,
    completed_snapshots: List[Dict[str, Any]],
) -> None:
    """Back-fill artifact_dir (and completed_attempt_id) for submitted members.

    Called after job(s) reach ARTIFACT_SUCCESS.  At cohort-write time the
    jobs have not yet run, so artifact_dir is null; this function patches the
    persisted cohort once the artifact directory is known.

    Matching priority (mirrors update_cohort_submitted):
      1. submitted_attempt_id == physical_attempt_id
      2. logical_run_id
    """
    cpath = cohort_path(output_dir, launch_id)
    if not cpath.exists():
        return
    try:
        with open(cpath, encoding="utf-8") as fh:
            cohort = json.load(fh)
    except Exception as exc:
        logger.warning(
            "update_cohort_artifact_dirs: could not read cohort at %s: %s", cpath, exc
        )
        return

    by_attempt: Dict[str, Dict[str, Any]] = {}
    by_lrid: Dict[str, Dict[str, Any]] = {}
    for snap in completed_snapshots:
        adir = snap.get("artifact_dir") or ""
        if not adir:
            continue
        attempt = str(snap.get("physical_attempt_id") or "")
        lrid = str(snap.get("logical_run_id") or "")
        if attempt:
            by_attempt[attempt] = snap
        if lrid:
            by_lrid[lrid] = snap

    updated = 0
    for member in cohort.get("members", []):
        if member.get("artifact_dir"):
            continue
        snap = by_attempt.get(member.get("submitted_attempt_id") or "") or by_lrid.get(
            member.get("logical_run_id") or ""
        )
        if not snap:
            continue
        adir = snap.get("artifact_dir") or ""
        if not adir:
            continue
        member["artifact_dir"] = adir
        if not member.get("completed_attempt_id"):
            member["completed_attempt_id"] = snap.get("physical_attempt_id") or None
        updated += 1

    if updated == 0:
        return
    try:
        _atomic_write_json(cpath, cohort)
    except Exception as exc:
        logger.warning(
            "update_cohort_artifact_dirs: could not write cohort at %s: %s", cpath, exc
        )


def write_latest_launch(
    *,
    output_dir: Path,
    launch_id: str,
    created_at: str,
) -> None:
    """Write .multiverse/latest_launch.json pointing to the new launch."""
    ldir = str(launch_dir(output_dir, launch_id))
    data: Dict[str, Any] = {
        "schema_version": LATEST_LAUNCH_SCHEMA_VERSION,
        "launch_id": launch_id,
        "launch_dir": ldir,
        "created_at": created_at,
    }
    _atomic_write_json(latest_launch_path(output_dir), data)


# ---------------------------------------------------------------------------
# Cohort update (submitted attempt IDs)  — Gap 3: stable identity matching
# ---------------------------------------------------------------------------


def update_cohort_submitted(
    *,
    output_dir: Path,
    launch_id: str,
    submitted_runs: List[Dict[str, Any]],
) -> None:
    """Patch cohort members with their submitted_attempt_id.

    ``submitted_runs`` is a list of dicts produced by ``SubmittedRun.to_dict()``
    optionally augmented with a ``member_id`` field (injected by the launcher for
    stable matching).

    Matching priority:
      1. ``member_id`` — injected by the caller; collision-proof.
      2. ``logical_run_id`` — stable across session loss.
      3. ``job_name`` — last-resort display-name fallback.

    Logs a warning if the cohort cannot be read or written so failures are
    observable rather than silent.
    """
    cpath = cohort_path(output_dir, launch_id)
    if not cpath.exists():
        logger.warning(
            "update_cohort_submitted: cohort not found at %s (launch_id=%s)",
            cpath, launch_id,
        )
        return
    try:
        with open(cpath, encoding="utf-8") as fh:
            cohort = json.load(fh)
    except Exception as exc:
        logger.warning(
            "update_cohort_submitted: could not read cohort at %s: %s", cpath, exc
        )
        return

    # Build lookup tables in priority order.
    by_member_id: Dict[str, str] = {}
    by_logical_run_id: Dict[str, str] = {}
    by_job_name: Dict[str, str] = {}
    for r in submitted_runs:
        attempt = r.get("attempt_id") or ""
        if not attempt:
            continue
        mid = r.get("member_id") or ""
        if mid:
            by_member_id[mid] = attempt
        lrid = r.get("logical_run_id") or ""
        if lrid:
            by_logical_run_id[lrid] = attempt
        jname = r.get("job_name") or ""
        if jname and jname not in by_job_name:
            by_job_name[jname] = attempt

    updated = 0
    for member in cohort.get("members", []):
        if member.get("submitted_attempt_id") or member.get("skipped"):
            continue
        attempt_id = (
            by_member_id.get(member.get("member_id") or "")
            or by_logical_run_id.get(member.get("logical_run_id") or "")
            or by_job_name.get(member.get("job_name") or "")
        )
        if attempt_id:
            member["submitted_attempt_id"] = attempt_id
            updated += 1

    if updated == 0:
        logger.warning(
            "update_cohort_submitted: no members updated for launch_id=%s "
            "(submitted %d runs, %d members in cohort)",
            launch_id, len(submitted_runs), len(cohort.get("members", [])),
        )

    try:
        _atomic_write_json(cpath, cohort)
    except Exception as exc:
        logger.warning(
            "update_cohort_submitted: could not write cohort at %s: %s", cpath, exc
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_latest_cohort(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the most-recently-written cohort for output_dir, or None."""
    lpath = latest_launch_path(output_dir)
    if not lpath.exists():
        return None
    try:
        with open(lpath, encoding="utf-8") as fh:
            latest = json.load(fh)
    except Exception:
        return None
    lid = latest.get("launch_id")
    if not lid:
        return None
    cpath = cohort_path(output_dir, lid)
    if not cpath.exists():
        return None
    try:
        with open(cpath, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Readiness resolution
# ---------------------------------------------------------------------------

STATUS_READY = "ready"
STATUS_RUNNING = "running"
STATUS_TRAINING_FAILED = "training_failed"
STATUS_CANCELLED = "cancelled"
STATUS_NOT_SUBMITTED = "not_submitted"
STATUS_MISSING_ARTIFACT_DIR = "missing_artifact_dir"
STATUS_BAD_ARTIFACT_MANIFEST = "bad_artifact_manifest"
STATUS_NO_EMBEDDINGS = "no_embeddings"
STATUS_MISSING_DATASET = "missing_dataset"
STATUS_UNSUPPORTED_DATASET = "unsupported_dataset"

_SUPPORTED_EXTENSIONS = {".h5ad", ".h5mu"}

_MVD_TERMINAL_SUCCESS = {"ARTIFACT_SUCCESS"}
_MVD_TERMINAL_FAILED = {
    "FAILED", "RECOVERY_PENDING", "EVALUATION_FAILED", "PROMOTION_FAILED"
}
_MVD_TERMINAL_CANCELLED = {"CANCELLED"}


def _verify_artifact_dir(artifact_dir: str) -> Tuple[str, str]:
    """Return (STATUS_READY, '') or (error_status, reason).

    Checks in order: directory exists → embeddings.h5 present → manifest
    sidecar integrity → full declared-output validation (Gap 7).  Embeddings
    presence is checked first so we always return STATUS_NO_EMBEDDINGS rather
    than STATUS_BAD_ARTIFACT_MANIFEST when the only problem is missing embeddings.
    h5py is lazy-imported by the validation module; environments without it skip
    the declared-output check after the existence check passes.
    """
    adir = Path(artifact_dir)
    if not adir.is_dir():
        return STATUS_MISSING_ARTIFACT_DIR, f"artifact directory not found: {artifact_dir}"

    # Fast existence check before any parsing — gives a precise status.
    if not (adir / "embeddings.h5").exists():
        return STATUS_NO_EMBEDDINGS, "embeddings.h5 not present in artifact directory"

    # Manifest integrity check (sidecar sha256 verification, no h5py needed).
    try:
        from ..artifact.manifest import read_manifest
        read_manifest(adir)
    except Exception as exc:
        return STATUS_BAD_ARTIFACT_MANIFEST, str(exc)

    # Full declared-output validation (Gap 7) — validates actual file content.
    try:
        from ..artifact.validation import (ModelOutputContract, ValidationLevel,
                                           validate_output_bundle)

        contract = ModelOutputContract.default(expected_n_obs=None)
        report = validate_output_bundle(adir, contract, level=ValidationLevel.BASIC)
        if not report.passed:
            reasons = "; ".join(i.message for i in report.refusals)
            emb_issue = any(
                "embedding" in i.message.lower() or "EMBEDDING" in i.code
                for i in report.refusals
            )
            if emb_issue:
                return STATUS_NO_EMBEDDINGS, reasons
            return STATUS_BAD_ARTIFACT_MANIFEST, reasons
    except ImportError:
        pass  # h5py not installed; existence check above already confirmed presence

    return STATUS_READY, ""


def _check_dataset(member: Dict[str, Any]) -> Tuple[str, str]:
    """Verify the dataset exists and has a supported extension.

    Tries the resolved (absolute) path stored at cohort-build time first, then
    falls back to the original path from the manifest/registry (Gap 6).
    """
    resolved = member.get("dataset_path_resolved") or ""
    original = member.get("dataset_path") or ""
    dataset_path = resolved or original
    if not dataset_path:
        return STATUS_MISSING_DATASET, "dataset_path not recorded in cohort"

    p = Path(dataset_path)
    if not p.exists() and resolved and original and resolved != original:
        # Try original as a last resort.
        p = Path(original)

    if not p.exists():
        return STATUS_MISSING_DATASET, f"dataset path not found: {dataset_path}"
    if p.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        return (
            STATUS_UNSUPPORTED_DATASET,
            f"unsupported dataset extension {p.suffix!r}; expected .h5ad or .h5mu",
        )
    return STATUS_READY, ""


def resolve_member_readiness(
    member: Dict[str, Any],
    *,
    mvd_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    completed_runs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Compute readiness for one cohort member.

    Returns a copy of ``member`` extended with ``readiness_status`` and
    ``readiness_reason``.

    ``mvd_snapshots`` maps attempt_id → kernel snapshot dict.
    ``completed_runs`` maps logical_run_id → {attempt_id, artifact_dir} from
    :func:`multiverse.runner.resume.completed_logical_runs` — used to revalidate
    skipped members (Gap 4) and for logical-run fallback after refresh (Gap 5).
    """
    result = dict(member)
    mvd_snapshots = mvd_snapshots or {}
    completed_runs = completed_runs or {}

    submitted_id = member.get("submitted_attempt_id")
    completed_id = member.get("completed_attempt_id")
    artifact_dir = member.get("artifact_dir")
    logical_run_id = member.get("logical_run_id") or ""

    # --- Branch: skipped_completed — revalidate through mvd (Gap 4) ---
    if member.get("skipped") and completed_id:
        if logical_run_id and completed_runs is not None:
            current = completed_runs.get(logical_run_id)
            if current is None:
                result["readiness_status"] = STATUS_MISSING_ARTIFACT_DIR
                result["readiness_reason"] = (
                    "skipped member's logical_run_id is no longer ARTIFACT_SUCCESS in mvd"
                )
                return result
            # Use the current artifact dir from the live index (may differ after rebuild).
            live_adir = current.get("artifact_dir") or artifact_dir
            if live_adir and live_adir != artifact_dir:
                result["artifact_dir"] = live_adir
                artifact_dir = live_adir
        if not artifact_dir:
            result["readiness_status"] = STATUS_MISSING_ARTIFACT_DIR
            result["readiness_reason"] = "skipped member has no recorded artifact_dir"
            return result
        status, reason = _verify_artifact_dir(artifact_dir)
        if status == STATUS_READY:
            status, reason = _check_dataset(member)
        result["readiness_status"] = status
        result["readiness_reason"] = reason
        return result

    # --- Branch: not submitted ---
    if not submitted_id and not completed_id:
        # Gap 5: try completed-run lookup by logical_run_id before giving up.
        if logical_run_id and completed_runs is not None:
            current = completed_runs.get(logical_run_id)
            if current:
                adir = current.get("artifact_dir") or ""
                result["artifact_dir"] = adir
                result["completed_attempt_id"] = current.get("attempt_id") or completed_id
                if adir:
                    status, reason = _verify_artifact_dir(adir)
                    if status == STATUS_READY:
                        status, reason = _check_dataset(member)
                    result["readiness_status"] = status
                    result["readiness_reason"] = reason
                    return result
        result["readiness_status"] = STATUS_NOT_SUBMITTED
        result["readiness_reason"] = "member was not submitted in this launch"
        return result

    # --- Branch: submitted — check mvd state ---
    attempt_id = submitted_id or completed_id
    snap = mvd_snapshots.get(str(attempt_id)) if attempt_id else None

    if snap is not None:
        primary_state = str(snap.get("primary_state") or "")

        if primary_state in _MVD_TERMINAL_SUCCESS:
            adir = snap.get("artifact_dir") or artifact_dir
            if not adir:
                result["readiness_status"] = STATUS_MISSING_ARTIFACT_DIR
                result["readiness_reason"] = "ARTIFACT_SUCCESS but no artifact_dir in snapshot"
                return result
            result["artifact_dir"] = adir
            status, reason = _verify_artifact_dir(adir)
            if status == STATUS_READY:
                status, reason = _check_dataset(member)
            result["readiness_status"] = status
            result["readiness_reason"] = reason
            return result

        if primary_state in _MVD_TERMINAL_FAILED:
            reason = snap.get("failure_reason") or ""
            result["readiness_status"] = STATUS_TRAINING_FAILED
            result["readiness_reason"] = f"run reached {primary_state}" + (
                f": {reason}" if reason else ""
            )
            return result

        if primary_state in _MVD_TERMINAL_CANCELLED:
            result["readiness_status"] = STATUS_CANCELLED
            result["readiness_reason"] = "run was cancelled"
            return result

        result["readiness_status"] = STATUS_RUNNING
        result["readiness_reason"] = f"run is in state {primary_state}"
        return result

    # No snapshot available: try artifact_dir from cohort, then logical-run lookup (Gap 5).
    if artifact_dir:
        status, reason = _verify_artifact_dir(artifact_dir)
        if status == STATUS_READY:
            status, reason = _check_dataset(member)
        result["readiness_status"] = status
        result["readiness_reason"] = reason
        return result

    if logical_run_id and completed_runs is not None:
        current = completed_runs.get(logical_run_id)
        if current:
            adir = current.get("artifact_dir") or ""
            result["artifact_dir"] = adir
            if adir:
                status, reason = _verify_artifact_dir(adir)
                if status == STATUS_READY:
                    status, reason = _check_dataset(member)
                result["readiness_status"] = status
                result["readiness_reason"] = reason
                return result

    result["readiness_status"] = STATUS_RUNNING
    result["readiness_reason"] = "submitted but no snapshot available yet"
    return result


def resolve_cohort_readiness(
    cohort: Dict[str, Any],
    *,
    mvd_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    completed_runs: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Return all members with readiness fields populated."""
    return [
        resolve_member_readiness(
            m, mvd_snapshots=mvd_snapshots, completed_runs=completed_runs
        )
        for m in cohort.get("members", [])
    ]


def readiness_summary(members_with_status: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate readiness counts across all members."""
    counts: Dict[str, int] = {}
    for m in members_with_status:
        s = m.get("readiness_status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    ready = counts.get(STATUS_READY, 0)
    total = len(members_with_status)
    return {
        "total": total,
        "ready": ready,
        "can_evaluate": ready > 0,
        "counts": counts,
    }


def filter_cohort_for_evaluation(
    cohort: Dict[str, Any],
    members_with_status: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a copy of ``cohort`` containing only the ready members.

    Uses the *resolved* member dicts from :func:`resolve_cohort_readiness` (not
    the raw cohort members) so that ``artifact_dir`` values back-filled during
    readiness resolution are carried into the config the evaluation container
    consumes. The container would otherwise fail on skipped or still-running
    members that have no embeddings to read.
    """
    ready = [
        m for m in members_with_status if m.get("readiness_status") == STATUS_READY
    ]
    filtered = dict(cohort)
    filtered["members"] = ready
    return filtered


# ---------------------------------------------------------------------------
# Pure rendering view-model (Gap 5 — testable without Streamlit)
# ---------------------------------------------------------------------------


def evaluate_section_view(
    members_with_status: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the pure view-model for the Evaluate Experiment section.

    This function contains no Streamlit calls and can be unit-tested directly.
    The GUI rendering function calls this and maps the result to Streamlit
    widgets.

    Returns a dict with:
    - ``summary_text``: human-readable readiness summary line.
    - ``table_rows``: list of dicts for the per-member table.
    - ``button_label``: label for the Evaluate button.
    - ``button_enabled``: whether the button should be active.
    - ``ready``: count of ready members.
    - ``total``: total member count.
    """
    summary = readiness_summary(members_with_status)
    ready = summary["ready"]
    total = summary["total"]

    non_ready_parts = [
        f"{count} {status.replace('_', ' ')}"
        for status, count in summary["counts"].items()
        if status != STATUS_READY and count > 0
    ]
    if non_ready_parts:
        summary_text = f"{ready}/{total} ready for evaluation ({', '.join(non_ready_parts)})"
    else:
        summary_text = f"{ready}/{total} ready for evaluation"

    table_rows = [
        {
            "Dataset": m.get("dataset_slug") or m.get("dataset_name") or "",
            "Model": m.get("model_slug") or "",
            "Source": m.get("source") or "",
            "Submitted attempt": m.get("submitted_attempt_id") or "",
            "Completed attempt": m.get("completed_attempt_id") or "",
            "Artifact dir": m.get("artifact_dir") or "",
            "Status": m.get("readiness_status") or "",
            "Reason": m.get("readiness_reason") or "",
        }
        for m in members_with_status
    ]

    button_label = (
        f"Evaluate experiment ({ready} ready)" if ready > 0 else "Evaluate experiment"
    )

    return {
        "summary_text": summary_text,
        "table_rows": table_rows,
        "button_label": button_label,
        "button_enabled": summary["can_evaluate"],
        "ready": ready,
        "total": total,
    }
