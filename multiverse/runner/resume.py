"""Opt-in, mvd-backed manifest resume (STRATEGY: MVD Manifest Resume and Dedupe).

Planning (``generate_execution_plan_from_manifest``) is a pure manifest → plan
expansion; it never decides that prior work makes a job unnecessary. *This*
module owns resume policy: given a plan and the mvd state root for the selected
output directory, it marks jobs whose canonical logical run already reached
``ARTIFACT_SUCCESS`` as ``_skipped`` — leaving them visible in the plan with a
reason and the completing attempt's provenance.

The legacy ``runs`` table is never consulted here. mvd completion state is
``ARTIFACT_SUCCESS`` recorded in the journal and projected into the rebuildable
SQLite index; legacy ``SUCCESS`` is not authoritative for mvd.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..index.sqlite_index import INDEX_FILENAME, open_index
from ..mvd import PrimaryState

SKIP_REASON = "completed logical run already has ARTIFACT_SUCCESS"


def resolve_skip_completed(
    *,
    cli_flag: Optional[bool] = None,
    manifest_data: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Resolve the effective skip-completed policy with documented precedence.

    1. an explicit CLI flag / GUI launch option (when not ``None``);
    2. ``globals.skip_completed`` in the manifest;
    3. ``False``.

    Keeping the default ``False`` means old manifests never start skipping work
    unexpectedly; resuming a large benchmark is always an intentional opt-in.
    """
    if cli_flag is not None:
        return bool(cli_flag)
    if manifest_data:
        globals_block = manifest_data.get("globals") or {}
        if isinstance(globals_block, Mapping) and "skip_completed" in globals_block:
            return bool(globals_block.get("skip_completed"))
    return False


def resolve_manifest_job_identity(
    job: Mapping[str, Any],
    *,
    manifest_hash: str,
    seed: Optional[int] = None,
    backend: str = "docker",
) -> str:
    """Compute the canonical mvd logical run id for a planned manifest job.

    This reuses the *exact* option-building and identity code that the executor
    runs, so a planned job and the attempt that completed it hash identically.
    There is no parallel ad-hoc key: the identity folds in manifest hash,
    dataset fingerprint, image identity, params hash, contract version, and the
    behaviour-affecting runtime fields (seed, preprocessing, model version) via
    :func:`multiverse.mvd...logical_run_id_for_spec`.
    """
    if backend == "slurm":
        from ..mvd.slurm_executor import (_resolve_identity_from_options,
                                          _SlurmJobSpec,
                                          logical_run_id_for_spec)
        from .mvd_entrypoint import _options_for_slurm_job

        options = _options_for_slurm_job(job, manifest_hash=manifest_hash, seed=seed)
        spec = _SlurmJobSpec.from_options(options, "resume-preflight")
    else:
        from ..mvd.docker_executor import (_ExecutorJobSpec,
                                           _resolve_identity_from_options,
                                           logical_run_id_for_spec)
        from .mvd_entrypoint import _options_for_job

        options = _options_for_job(job, manifest_hash=manifest_hash, seed=seed)
        spec = _ExecutorJobSpec.from_options(options, "resume-preflight")

    identity = _resolve_identity_from_options(spec)
    return logical_run_id_for_spec(spec, identity)


def completed_logical_runs(state_root: Path) -> Dict[str, Dict[str, str]]:
    """Map ``logical_run_id -> {attempt_id, artifact_dir}`` for completed work.

    Only ``ARTIFACT_SUCCESS`` attempts with an artifact directory that still
    exists on disk count as completed. ``FAILED``, ``CANCELLED``,
    ``RECOVERY_PENDING`` and any other state never appear here, so they cannot
    suppress a job. Reads the rebuildable SQLite index first and falls back to
    replaying the journal if the index is missing or unreadable.
    """
    state_root = Path(state_root)
    success = PrimaryState.ARTIFACT_SUCCESS.value
    out: Dict[str, Dict[str, str]] = {}

    index_path = state_root / INDEX_FILENAME
    if index_path.exists():
        try:
            with open_index(index_path, create_if_missing=False) as index:
                for row in index.list_runs(primary_state=success):
                    _record_completion(out, row)
            return out
        except Exception:
            out.clear()

    try:
        from .mvd_inprocess import snapshots_from_journal

        for snap in snapshots_from_journal(state_root=state_root, state=success):
            _record_completion(out, snap)
    except Exception:
        return out
    return out


def _record_completion(out: Dict[str, Dict[str, str]], row: Mapping[str, Any]) -> None:
    logical_run_id = row.get("logical_run_id")
    artifact_dir = row.get("artifact_dir")
    if not logical_run_id or not artifact_dir:
        return
    if not Path(artifact_dir).is_dir():
        return
    out[str(logical_run_id)] = {
        "attempt_id": str(row.get("physical_attempt_id") or ""),
        "artifact_dir": str(artifact_dir),
    }


def decorate_plan_with_resume(
    plan: List[Dict[str, Any]],
    *,
    state_root: Path,
    manifest_hash: str,
    seed: Optional[int] = None,
    backend: str = "docker",
) -> List[Dict[str, Any]]:
    """Return a copy of ``plan`` where completed jobs are marked ``_skipped``.

    Skipped jobs remain present (never silently dropped) and carry the
    completing attempt id and artifact directory so callers can show
    provenance. Jobs already marked ``_skipped`` (e.g. pre-flight validation
    failures) are passed through untouched. Every runnable job is annotated with
    its resolved ``_logical_run_id`` for diagnostics.
    """
    completed = completed_logical_runs(state_root)
    decorated: List[Dict[str, Any]] = []
    for job in plan:
        if job.get("_skipped"):
            decorated.append(job)
            continue
        try:
            logical_run_id = resolve_manifest_job_identity(
                job, manifest_hash=manifest_hash, seed=seed, backend=backend
            )
        except Exception:
            # An un-hashable job (missing options) cannot be matched against
            # completed work; leave it runnable rather than guessing.
            decorated.append(job)
            continue
        match = completed.get(logical_run_id)
        if match:
            decorated.append(
                {
                    **job,
                    "_skipped": True,
                    "_skip_reason": SKIP_REASON,
                    "_completed_attempt_id": match["attempt_id"],
                    "_completed_artifact_dir": match["artifact_dir"],
                    "_logical_run_id": logical_run_id,
                }
            )
        else:
            decorated.append({**job, "_logical_run_id": logical_run_id})
    return decorated
