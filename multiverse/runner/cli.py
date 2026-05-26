import argparse
import hashlib
import json
import os
import asyncio
import signal
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from rich.live import Live
from rich.table import Table
from rich.console import Console

try:
    from .docker_runner import (
        run_model_container,
        run_evaluation_container,
        build_images_concurrently,
        run_jobs_concurrently,
        ensure_docker_data_root,
        _supervise_container,
        mark_active_runs_failed_direct,
        start_db_writer,
        stop_db_writer,
    )
except ImportError as exc:
    _DOCKER_RUNNER_IMPORT_ERROR = exc

    def _raise_docker_unavailable(*args, **kwargs):
        raise RuntimeError(
            "Docker execution requires the docker Python package. "
            "Install the Docker dependencies or run with --local."
        ) from _DOCKER_RUNNER_IMPORT_ERROR

    def run_model_container(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    def run_evaluation_container(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    async def build_images_concurrently(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    async def run_jobs_concurrently(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    def ensure_docker_data_root(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    async def _supervise_container(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    def mark_active_runs_failed_direct(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    def start_db_writer(*args, **kwargs):
        return _raise_docker_unavailable(*args, **kwargs)

    async def stop_db_writer(*args, **kwargs):
        return None
from ..logging_utils import get_logger, setup_logging
from ..ingestion import register_from_manifest, resolve_manifest_path, preprocess_dataset
from ..registry_db import init_db, get_db_connection, ARTIFACTS_DIR, recover_orphaned_runs
from ..models_ingest import (
    resolve_model_manifest_path,
    load_model_manifest,
    register_model_from_manifest,
)
try:
    from ..builder import build_local_model
except ImportError as exc:
    _BUILDER_IMPORT_ERROR = exc

    def build_local_model(*args, **kwargs):
        raise RuntimeError(
            "Model image builds require the docker Python package. "
            "Install the Docker dependencies before running build commands."
        ) from _BUILDER_IMPORT_ERROR
import yaml

logger = get_logger(__name__)
console = Console()


def emit_event(event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)

# Required omics per model slug. Empty set means any combination is acceptable.


@dataclass
class ParsedManifest:
    path: Path
    data: Dict[str, Any] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class ManifestValidationError(ValueError):
    def __init__(self, parsed: ParsedManifest):
        self.parsed = parsed
        message = "; ".join(f"{e['field']}: {e['message']}" for e in parsed.errors)
        super().__init__(message or "manifest validation failed")


def _manifest_error(field: str, message: str, code: str = "invalid") -> Dict[str, str]:
    return {"field": field, "message": message, "code": code}

MODEL_REQUIRED_OMICS = {
    "multivi": {"rna", "atac"},
    "totalvi": {"rna", "adt"},
    "cobolt": {"rna", "atac"},
    "mofa": set(),
    "mowgli": set(),
    "pca": set(),
}


@dataclass
class _PeekResult:
    value: Any
    error: str | None = None


def _import_h5py():
    """Lazy import so control-plane commands don't require ML deps."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for dataset content validation during run planning. "
            "Install with the ml-legacy dependency group."
        ) from exc
    return h5py


def _peek_obs_columns(path: str) -> _PeekResult:
    """Return obs columns plus a structured error if the HDF5 header is unreadable.

    For .h5mu files, columns from all modality-level obs groups are merged with the
    top-level obs so that keys stored only under /mod/*/obs are not falsely reported
    as absent during preflight validation.
    """
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            cols: set = set()
            obs = f.get("obs")
            if obs is not None:
                cols |= {k for k in obs.keys() if not k.startswith("_")}
            mod = f.get("mod")
            if mod is not None:
                for mod_name in mod.keys():
                    mod_obs = mod[mod_name].get("obs")
                    if mod_obs is not None:
                        cols |= {k for k in mod_obs.keys() if not k.startswith("_")}
            if not cols:
                return _PeekResult(set(), "missing_obs")
            return _PeekResult(cols)
    except PermissionError as exc:
        logger.debug(f"Permission denied reading obs columns from {path}: {exc}")
        return _PeekResult(set(), "permission_denied")
    except (OSError, KeyError, ValueError) as exc:
        logger.debug(f"Could not read obs columns from {path}: {exc}")
        return _PeekResult(set(), "file_unreadable")


def _read_obs_columns(path: str) -> set:
    """Return the set of obs column names from an h5ad or h5mu file via h5py."""
    result = _peek_obs_columns(path)
    return result.value if isinstance(result.value, set) else set()


def _peek_obs_count(path: str) -> _PeekResult:
    """Return n_obs from obs/_index, or first obs column, without loading X."""
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            obs = f.get("obs")
            if obs is None:
                return _PeekResult(0, "missing_obs")
            if "_index" in obs:
                return _PeekResult(int(obs["_index"].shape[0]))
            for key in obs.keys():
                node = obs[key]
                if hasattr(node, "shape"):
                    return _PeekResult(int(node.shape[0]))
                if "codes" in node:
                    return _PeekResult(int(node["codes"].shape[0]))
            return _PeekResult(0, "zero_cells")
    except PermissionError as exc:
        logger.debug(f"Permission denied reading obs count from {path}: {exc}")
        return _PeekResult(0, "permission_denied")
    except (OSError, KeyError, ValueError) as exc:
        logger.debug(f"Could not read obs count from {path}: {exc}")
        return _PeekResult(0, "file_unreadable")


def _read_obs_count(path: str) -> int:
    result = _peek_obs_count(path)
    return int(result.value or 0)


def _peek_batch_count(path: str, batch_key: str) -> _PeekResult:
    """Return number of unique batch values plus structured error if unreadable.

    For .h5mu files, modality-level obs groups are checked when the key is
    absent from top-level obs, consistent with _peek_obs_columns behaviour.
    """
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            def _count_from_node(batch_node) -> int:
                if "categories" in batch_node:
                    return int(batch_node["categories"].shape[0])
                return int(len(set(batch_node[:])))

            obs = f.get("obs")
            if obs is not None and batch_key in obs:
                return _PeekResult(_count_from_node(obs[batch_key]))

            mod = f.get("mod")
            if mod is not None:
                for mod_name in mod.keys():
                    mod_obs = mod[mod_name].get("obs")
                    if mod_obs is not None and batch_key in mod_obs:
                        return _PeekResult(_count_from_node(mod_obs[batch_key]))

            return _PeekResult(0, "missing_batch_key")
    except PermissionError as exc:
        logger.debug(f"Permission denied reading batch count from {path}: {exc}")
        return _PeekResult(0, "permission_denied")
    except (OSError, KeyError, ValueError) as exc:
        logger.debug(f"Could not read batch count from {path}: {exc}")
        return _PeekResult(0, "file_unreadable")


def _read_batch_count(path: str, batch_key: str) -> int:
    """Return number of unique batch values. 0 if unreadable."""
    result = _peek_batch_count(path, batch_key)
    return int(result.value or 0)


def _record_validation_failure(job: Dict[str, Any], reason: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO runs
            (dataset_id, model_slug, model_version, model_name, status, output_path, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.get("dataset_id"),
                job.get("model_slug", job.get("model_name", "")),
                job.get("model_version", "0.0.0"),
                job.get("model_name", job.get("model_slug", "")),
                "FAILED",
                job.get("output_path", ""),
                f"VALIDATION_ERROR:{reason}",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def validate_pending_jobs(
    pending_jobs: List[Dict],
    record_failures: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """Pre-flight validation gate.

    Checks each job for modality compatibility, batch key presence, and
    cell type key presence. Returns (validated_jobs, warnings). Incompatible
    jobs are silently dropped with a logged reason; warnings are surfaced in
    the end-of-run summary.
    """
    warnings: List[str] = []
    validated: List[Dict] = []

    # Cache per dataset_id so each file is opened at most once.
    obs_cache: Dict[int, set] = {}
    obs_error_cache: Dict[int, str | None] = {}
    obs_count_cache: Dict[int, _PeekResult] = {}
    batch_count_cache: Dict[int, int] = {}

    def skip_job(job: Dict[str, Any], reason: str, message: str) -> None:
        logger.warning(message)
        if record_failures:
            _record_validation_failure(job, reason)
        validated.append({**job, "_skipped": True, "_skip_reason": message})

    for job in pending_jobs:
        dataset_id = job["dataset_id"]
        dataset_name = job["dataset_name"]
        dataset_path = job["dataset_path"]
        model_slug = job.get("model_slug", job.get("model_name", ""))
        omics_available = set(job.get("omics_available") or [])
        batch_key = job.get("batch_key")
        cell_type_key = job.get("cell_type_key")

        # 1. Required omics check
        required = MODEL_REQUIRED_OMICS.get(model_slug, set())
        if required and not required.issubset(omics_available):
            skip_job(
                job,
                "missing_required_omics",
                f"[SKIP] {dataset_name}/{model_slug}: model requires omics {required}, "
                f"dataset has {omics_available}",
            )
            continue

        if dataset_id not in obs_count_cache:
            obs_count_cache[dataset_id] = _peek_obs_count(dataset_path)
        obs_count = obs_count_cache[dataset_id]
        if obs_count.error in {"file_unreadable", "permission_denied"}:
            skip_job(job, obs_count.error, f"[SKIP] {dataset_name}/{model_slug}: dataset file unreadable ({obs_count.error})")
            continue
        if int(obs_count.value or 0) == 0:
            skip_job(job, "zero_cells", f"[SKIP] {dataset_name}/{model_slug}: dataset has zero cells")
            continue

        # 2. Batch key presence check (only if registry declares a batch_key)
        if batch_key:
            if dataset_id not in obs_cache:
                obs_cache[dataset_id] = _read_obs_columns(dataset_path)
                if obs_cache[dataset_id]:
                    obs_error_cache[dataset_id] = None
                else:
                    obs_error_cache[dataset_id] = _peek_obs_columns(dataset_path).error
            obs_cols = obs_cache[dataset_id]
            obs_error = obs_error_cache.get(dataset_id)
            if obs_error in {"file_unreadable", "permission_denied"}:
                skip_job(job, obs_error, f"[SKIP] {dataset_name}/{model_slug}: dataset obs unreadable ({obs_error})")
                continue
            if batch_key not in obs_cols:
                skip_job(
                    job,
                    "missing_batch_key",
                    f"[SKIP] {dataset_name}/{model_slug}: batch_key '{batch_key}' not found in dataset obs columns",
                )
                continue

        # 3. Cell type key warning (don't skip)
        if cell_type_key:
            if dataset_id not in obs_cache:
                obs_cache[dataset_id] = _read_obs_columns(dataset_path)
                if obs_cache[dataset_id]:
                    obs_error_cache[dataset_id] = None
                else:
                    obs_error_cache[dataset_id] = _peek_obs_columns(dataset_path).error
            obs_cols = obs_cache[dataset_id]
            if cell_type_key not in obs_cols:
                msg = (
                    f"dataset '{dataset_name}': cell_type_key '{cell_type_key}' not found "
                    f"— supervised metrics will be skipped"
                )
                logger.warning(f"[WARN] {msg}")
                warnings.append(msg)

        # 4. Single batch warning (don't skip)
        if batch_key:
            if dataset_id not in batch_count_cache:
                batch_count_cache[dataset_id] = _read_batch_count(dataset_path, batch_key)
            n_batches = batch_count_cache[dataset_id]
            if n_batches == 1:
                msg = (
                    f"dataset '{dataset_name}': only 1 batch value found for '{batch_key}' "
                    f"— batch-correction metrics will be skipped"
                )
                logger.warning(f"[WARN] {msg}")
                if msg not in warnings:
                    warnings.append(msg)

        validated.append(job)

    return validated, warnings


def parse_manifest(manifest_path: str, db_conn) -> ParsedManifest:
    """Parse, validate, and dry-run a run manifest against the live registry."""
    path = Path(manifest_path).expanduser().resolve()
    parsed = ParsedManifest(path=path)

    try:
        with open(path, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp)
    except FileNotFoundError:
        parsed.errors.append(_manifest_error("manifest", f"file not found: {path}", "file_not_found"))
        return parsed
    except UnicodeDecodeError as exc:
        parsed.errors.append(_manifest_error("manifest", f"file is not valid UTF-8: {exc}", "unicode_error"))
        return parsed
    except yaml.YAMLError as exc:
        parsed.errors.append(_manifest_error("manifest", f"YAML syntax error: {exc}", "yaml_error"))
        return parsed

    if not isinstance(loaded, dict):
        parsed.errors.append(_manifest_error("manifest", "top-level document must be a mapping", "schema_error"))
        return parsed
    parsed.data = loaded

    jobs = loaded.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        parsed.errors.append(_manifest_error("jobs", "manifest must contain at least one job", "schema_error"))
        return parsed

    cursor = db_conn.cursor()
    for idx, job in enumerate(jobs):
        field_prefix = f"jobs[{idx}]"
        if not isinstance(job, dict):
            parsed.errors.append(_manifest_error(field_prefix, "job must be a mapping", "schema_error"))
            continue

        dataset_key = job.get("dataset_slug") or job.get("dataset_id")
        if not dataset_key:
            parsed.errors.append(_manifest_error(f"{field_prefix}.dataset_slug", "dataset_slug or dataset_id is required", "schema_error"))
        else:
            cursor.execute(
                "SELECT status FROM datasets WHERE (slug = ? OR name = ?) LIMIT 1",
                (dataset_key, dataset_key),
            )
            row = cursor.fetchone()
            if row is None:
                parsed.errors.append(_manifest_error(f"{field_prefix}.dataset_slug", f"dataset '{dataset_key}' is not registered", "stale_dataset_slug"))
            elif row[0] != "READY":
                parsed.errors.append(_manifest_error(f"{field_prefix}.dataset_slug", f"dataset '{dataset_key}' is {row[0]}, not READY", "dataset_not_ready"))

        if isinstance(job.get("models"), list):
            models_to_run = list(job.get("models", []))
        elif job.get("model_name"):
            models_to_run = [job.get("model_name")]
        else:
            models_to_run = []
        if not models_to_run:
            parsed.errors.append(_manifest_error(f"{field_prefix}.models", "models or model_name is required", "schema_error"))
        for model_name in models_to_run:
            model_key = str(model_name).strip().lower()
            cursor.execute(
                "SELECT status FROM models WHERE slug = ? ORDER BY version DESC LIMIT 1",
                (model_key,),
            )
            row = cursor.fetchone()
            if row is None:
                parsed.errors.append(_manifest_error(f"{field_prefix}.models", f"model '{model_name}' is not registered", "stale_model_slug"))
            elif row[0] != "ACTIVE":
                parsed.errors.append(_manifest_error(f"{field_prefix}.models", f"model '{model_name}' is {row[0]}, not ACTIVE", "model_not_active"))

    if parsed.errors:
        return parsed

    parsed.plan = generate_execution_plan_from_manifest(db_conn, parsed.data)
    if not parsed.plan:
        parsed.errors.append(_manifest_error("jobs", "manifest dry-run produced no runnable jobs", "empty_plan"))
    return parsed


def require_parsed_manifest(manifest_path: str, db_conn) -> ParsedManifest:
    parsed = parse_manifest(manifest_path, db_conn)
    if not parsed.ok:
        raise ManifestValidationError(parsed)
    return parsed


def generate_execution_plan_from_manifest(conn, manifest_data: Dict) -> List[Dict]:
    """Generates a list of required ML jobs based on a manifest."""
    cursor = conn.cursor()
    pending_jobs = []
    global_metrics = manifest_data.get("globals", {}).get("metrics", {})

    for job in manifest_data.get("jobs", []):
        dataset_key = job.get("dataset_slug") or job.get("dataset_id")
        models_to_run = []
        if isinstance(job.get("models"), list):
            models_to_run = list(job.get("models", []))
        elif job.get("model_name"):
            models_to_run = [job.get("model_name")]
        mode = job.get("mode", "run")

        cursor.execute(
            "SELECT id, name, path, omics_available, batch_key, cell_type_key "
            "FROM datasets WHERE (slug = ? OR name = ?) AND status = 'READY'",
            (dataset_key, dataset_key),
        )
        dataset_row = cursor.fetchone()
        if not dataset_row:
            logger.warning(f"Dataset '{dataset_key}' not found in registry or not READY. Skipping.")
            continue

        d_id, d_name, d_path, d_omics_json, d_batch_key, d_cell_type_key = dataset_row
        d_omics = json.loads(d_omics_json) if d_omics_json else []

        for m_name in models_to_run:
            model_lookup = str(m_name).strip().lower()
            cursor.execute(
                """
                SELECT docker_image, slug, version
                FROM models
                WHERE slug = ? AND status = 'ACTIVE'
                ORDER BY version DESC
                LIMIT 1
                """,
                (model_lookup,),
            )
            model_row = cursor.fetchone()
            if not model_row:
                logger.warning(f"Model '{m_name}' not found in registry. Skipping.")
                continue

            m_image, m_slug, m_version = model_row

            # Manifest-driven runs always execute — the user explicitly requested
            # them. Deduplication (skip prior-SUCCESS) applies only to the
            # headless registry-sweep path (generate_execution_plan).
            job_metrics = job.get("metrics", {})
            merged_metrics = {**global_metrics, **job_metrics}

            model_params = job.get("model_params", {}) or {}
            params_hash = hashlib.sha256(
                json.dumps(model_params, sort_keys=True).encode()
            ).hexdigest()[:12]

            output_path = os.path.join(ARTIFACTS_DIR, f"{d_name}_{m_slug}")
            job_entry = {
                "dataset_id": d_id,
                "dataset_name": d_name,
                "dataset_path": d_path,
                "omics_available": d_omics,
                "batch_key": d_batch_key,
                "cell_type_key": d_cell_type_key,
                "model_name": m_slug,
                "model_slug": m_slug,
                "model_version": m_version,
                "model_image": m_image,
                "output_path": output_path,
                "mode": mode,
                "model_params": model_params,
                "params_hash": params_hash,
                "search_space": job.get("search_space", {}),
                "optimize_metric": job.get("optimize_metric"),
                "n_trials": job.get("n_trials", 10),
                "direction": job.get("direction", "maximize"),
                "study_storage": job.get("study_storage", "sqlite:///optuna.db"),
                "metrics": merged_metrics,
            }
            if job.get("mem_limit"):
                job_entry["mem_limit"] = job["mem_limit"]
            pending_jobs.append(job_entry)

    return pending_jobs


def generate_execution_plan(conn) -> List[Dict]:
    """Generates a list of required ML jobs by checking the database."""
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, path, omics_available, batch_key, cell_type_key "
        "FROM datasets WHERE status = 'READY'"
    )
    datasets = cursor.fetchall()

    cursor.execute(
        "SELECT slug, version, docker_image, supported_omics FROM models WHERE status = 'ACTIVE'"
    )
    models = cursor.fetchall()

    pending_jobs = []

    for d_id, d_name, d_path, d_omics_json, d_batch_key, d_cell_type_key in datasets:
        d_omics = set(json.loads(d_omics_json))

        for m_slug, m_version, m_image, m_omics_json in models:
            m_omics = set(json.loads(m_omics_json))

            if m_omics.issubset(d_omics):
                cursor.execute(
                    "SELECT status FROM runs WHERE dataset_id = ? AND model_slug = ? AND model_version = ?",
                    (d_id, m_slug, m_version)
                )
                run = cursor.fetchone()

                if run is None or run[0] != 'SUCCESS':
                    output_path = os.path.join(ARTIFACTS_DIR, f"{d_name}_{m_slug}")
                    pending_jobs.append({
                        "dataset_id": d_id,
                        "dataset_name": d_name,
                        "dataset_path": d_path,
                        "omics_available": list(d_omics),
                        "batch_key": d_batch_key,
                        "cell_type_key": d_cell_type_key,
                        "model_name": m_slug,
                        "model_slug": m_slug,
                        "model_version": m_version,
                        "model_image": m_image,
                        "output_path": output_path,
                        "metrics": {},
                    })

    return pending_jobs


def generate_status_table(tasks: Dict[str, str]) -> Table:
    """Generates a Rich Table representing the current status of all tasks."""
    table = Table(title="Multiverse Parallel Execution Dashboard")
    table.add_column("Task/Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", style="magenta")

    for name, status in tasks.items():
        style = "green" if status in ["Success", "Ready"] else "red" if "Failed" in status or "Error" in status else "yellow"
        table.add_row(name, f"[{style}]{status}[/]")

    return table


def _print_run_summary(
    all_jobs: List[Dict],
    result_summary: Dict[str, str],
    warnings: List[str],
) -> None:
    """Print an end-of-run Rich table with per-job status and overall counts."""
    table = Table(title="Run Summary", show_header=True)
    table.add_column("Job", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="blue")
    table.add_column("Model", style="blue")
    table.add_column("Status", justify="center")
    table.add_column("Note", style="dim")

    n_success = n_failed = n_skipped = 0

    for job in all_jobs:
        job_name = job.get("name") or f"{job.get('dataset_name', '?')}_{job.get('model_slug') or job.get('model_name', '?')}"
        dataset = job.get("dataset_name", "?")
        model = job.get("model_slug") or job.get("model_name", "?")

        if job.get("_skipped"):
            status_str = "[yellow]SKIPPED[/]"
            note = job.get("_skip_reason", "")
            n_skipped += 1
        else:
            outcome = result_summary.get(job_name, "unknown")
            if outcome == "success":
                status_str = "[green]SUCCESS[/]"
                note = ""
                n_success += 1
            else:
                status_str = "[red]FAILED[/]"
                note = "see workspace logs"
                n_failed += 1

        table.add_row(job_name, dataset, model, status_str, note)

    console.print(table)
    console.print(
        f"Completed: [green]{n_success} succeeded[/], "
        f"[red]{n_failed} failed[/], "
        f"[yellow]{n_skipped} skipped[/] of {len(all_jobs)} jobs."
    )
    if warnings:
        console.print("\n[bold yellow]Warnings:[/]")
        for w in warnings:
            console.print(f"  [yellow]•[/] {w}")


async def run_workflow_async(args: argparse.Namespace):
    """Executes the concurrent Docker-based workflow with pre-flight validation."""
    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output)

    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()
    previous_handlers: dict[int, Any] = {}
    reattach_specs: List[Tuple[Any, int, str, str]] = []

    def _request_shutdown() -> None:
        logger.warning("Shutdown requested; cancelling active workflow tasks")
        try:
            marked = mark_active_runs_failed_direct("CANCELLED")
            if marked:
                emit_event("status", job="orchestrator", status="marked_cancelled", count=marked)
        except Exception as exc:
            emit_event("error", kind="direct_write_failed", message=str(exc))
        if current_task is not None:
            current_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            loop.add_signal_handler(sig, _request_shutdown)
        except (NotImplementedError, RuntimeError):
            previous_handlers[sig] = signal.signal(sig, lambda _sig, _frame: _request_shutdown())

    writer_started = False
    conn = None
    try:
        docker_client = None
        docker_unavailable: Exception | None = None
        try:
            import docker  # type: ignore
            docker_client = docker.from_env()
            try:
                docker_client.ping()
            except Exception as exc:
                emit_event("error", kind="docker_daemon_offline", message=str(exc))
                raise RuntimeError(f"Docker daemon is not reachable: {exc}") from exc
        except Exception as exc:
            docker_unavailable = exc
            logger.warning("Docker client unavailable during recovery: %s", exc)
            emit_event("error", kind="docker_client_unavailable", message=str(exc))

        def _collect_reattach(container, run_id, output_path):
            labels = ((getattr(container, "attrs", {}) or {}).get("Config", {}) or {}).get("Labels", {}) or {}
            workspace_dir = labels.get("multiverse.run.workspace_dir") or output_path
            final_artifact_dir = labels.get("multiverse.run.final_artifact_dir") or output_path
            reattach_specs.append((container, int(run_id), str(workspace_dir), str(final_artifact_dir)))

        recover_orphaned_runs(docker_client=docker_client, reattach_callback=_collect_reattach)

        conn = get_db_connection()
        manifest_data: Dict[str, Any] = {}
        manifest_run_id: str | None = None
        if hasattr(args, "manifest") and args.manifest:
            parsed_manifest = require_parsed_manifest(args.manifest, conn)
            manifest_data = parsed_manifest.data
            pending_jobs = parsed_manifest.plan
            # Stable ID for this entire manifest invocation — used to group all
            # runs from one execution together and disambiguate re-runs of the
            # same manifest. Pin via globals.manifest_run_id for reproducible CI.
            manifest_run_id = (
                manifest_data.get("globals", {}).get("manifest_run_id")
                or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
            )
            for job in pending_jobs:
                job["manifest_run_id"] = manifest_run_id
            logger.info("Manifest run ID: %s", manifest_run_id)
        else:
            pending_jobs = generate_execution_plan(conn)
        conn.close()
        conn = None

        host_ram_gb: float | None = manifest_data.get("globals", {}).get("host_ram_gb")

        if not pending_jobs and not reattach_specs:
            logger.info("No pending jobs to execute.")
            return
        if docker_unavailable is not None:
            raise RuntimeError(f"Docker is unavailable; refusing to queue jobs: {docker_unavailable}") from docker_unavailable

        # Start the single-writer DB actor only after planning/recovery succeeds. All DB
        # writes from parallel worker coroutines are funnelled through this task.
        start_db_writer()
        writer_started = True

        reattach_tasks = [
            asyncio.create_task(
                _supervise_container(container, run_id, workspace_dir, final_artifact_dir),
                name=f"reattach_run_{run_id}",
            )
            for container, run_id, workspace_dir, final_artifact_dir in reattach_specs
        ]

        if not pending_jobs:
            if reattach_tasks:
                await asyncio.gather(*reattach_tasks)
            return

        validated_jobs, pre_flight_warnings = validate_pending_jobs(pending_jobs, record_failures=True)
        runnable_jobs = [j for j in validated_jobs if not j.get("_skipped")]

        if not runnable_jobs:
            logger.info("All jobs were skipped after pre-flight validation.")
            _print_run_summary(validated_jobs, {}, pre_flight_warnings)
            return

        experiment_name = (
            manifest_data.get("globals", {}).get("experiment_name")
            or (os.path.basename(os.path.normpath(args.output)) if args.output else None)
            or "default_experiment"
        )
        models_info = []
        for job in runnable_jobs:
            job_entry = {
                "name": f"{job['dataset_name']}_{job['model_name']}",
                "image": job['model_image'],
                "dataset_path": job['dataset_path'],
                "output_path": job['output_path'],
                "dataset_id": job['dataset_id'],
                "model_name_orig": job['model_name'],
                "model_slug": job.get("model_slug", job["model_name"]),
                "model_version": job.get("model_version", "0.0.0"),
                "experiment_name": experiment_name,
                "mode": job.get("mode", "run"),
                "model_params": job.get("model_params", {}) or {},
                "search_space": job.get("search_space", {}),
                "optimize_metric": job.get("optimize_metric"),
                "n_trials": job.get("n_trials", 10),
                "direction": job.get("direction", "maximize"),
                "study_storage": job.get("study_storage", "sqlite:///optuna.db"),
                "metrics": job.get("metrics", {}),
                "manifest_run_id": job.get("manifest_run_id"),
                "params_hash": job.get("params_hash"),
            }
            if job.get("mem_limit"):
                job_entry["mem_limit"] = job["mem_limit"]
            models_info.append(job_entry)

        tasks_status = {m["name"]: "Pending" for m in models_info}
        image_tags = list(set(m["image"] for m in models_info))
        for tag in image_tags:
            tasks_status[tag] = "Queued"

        result_summary: Dict[str, str] = {}

        ensure_docker_data_root()
        from ..multiverse_config import get_docker_data_root
        console.print(f"[bold]Docker data root:[/bold] {get_docker_data_root()}")

        with Live(generate_status_table(tasks_status), refresh_per_second=4) as live:
            def update_status(name, status):
                tasks_status[name] = status
                if status in {"Starting", "Running"} or "In Progress" in str(status):
                    emit_event("job_start", job=name, status=status)
                elif status in {"Success", "Ready"} or "Success" in str(status):
                    emit_event("job_end", job=name, status="SUCCESS")
                elif "Failed" in str(status) or "Error" in str(status):
                    emit_event("job_end", job=name, status="FAILED", message=str(status))
                else:
                    emit_event("status", job=name, status=status)
                live.update(generate_status_table(tasks_status))

            update_status("Image Preparation", "In Progress")
            try:
                await build_images_concurrently(image_tags, status_callback=update_status)
                update_status("Image Preparation", "Ready")
            except Exception as e:
                update_status("Image Preparation", f"Failed: {e}")
                _print_run_summary(validated_jobs, {}, pre_flight_warnings)
                return

            update_status("Model Execution", "In Progress")

            sweep_jobs = [m for m in models_info if m.get("mode") == "sweep"]
            run_jobs = [m for m in models_info if m.get("mode") != "sweep"]

            if run_jobs:
                result_summary.update(
                    await run_jobs_concurrently(
                        run_jobs,
                        args.seed,
                        status_callback=update_status,
                        host_ram_gb=host_ram_gb,
                    )
                )
            if sweep_jobs:
                from .tuner import run_sweep

                for sweep_job in sweep_jobs:
                    try:
                        result = run_sweep({
                            "name": sweep_job["name"],
                            "image": sweep_job["image"],
                            "dataset_id": sweep_job["dataset_id"],
                            "dataset_name": sweep_job["dataset_name"],
                            "dataset_path": sweep_job["dataset_path"],
                            "model_name_orig": sweep_job["model_name_orig"],
                            "model_slug": sweep_job.get("model_slug", sweep_job["model_name_orig"]),
                            "model_version": sweep_job.get("model_version", "0.0.0"),
                            "experiment_name": sweep_job.get("experiment_name", "default_experiment"),
                            "model_params": sweep_job.get("model_params", {}) or {},
                            "search_space": sweep_job.get("search_space", {}),
                            "optimize_metric": sweep_job.get("optimize_metric"),
                            "n_trials": sweep_job.get("n_trials", 10),
                            "direction": sweep_job.get("direction", "maximize"),
                            "study_storage": sweep_job.get("study_storage", "sqlite:///optuna.db"),
                            "seed": args.seed,
                        })
                        logger.info(
                            f"Sweep completed for {sweep_job['name']} best={result['best_value']} params={result['best_params']}"
                        )
                        result_summary[sweep_job["name"]] = "success"
                        update_status(sweep_job["name"], "Success")
                    except Exception as exc:
                        logger.error(f"Sweep failed for {sweep_job['name']}: {exc}")
                        result_summary[sweep_job["name"]] = "failed"
                        update_status(sweep_job["name"], "Failed")

            failed_models = [name for name, status in result_summary.items() if status == "failed"]
            if failed_models:
                update_status("Model Execution", f"Completed with failures: {failed_models}")
            else:
                update_status("Model Execution", "Success")

        if reattach_tasks:
            reattach_results = await asyncio.gather(*reattach_tasks, return_exceptions=True)
            for result in reattach_results:
                if isinstance(result, Exception):
                    logger.error("Recovered container supervision failed: %s", result)

        _print_run_summary(validated_jobs, result_summary, pre_flight_warnings)
    except asyncio.CancelledError:
        logger.warning("Workflow cancelled; DB writer will be drained before exit")
        raise
    finally:
        if conn is not None:
            conn.close()
        if writer_started:
            await asyncio.shield(stop_db_writer())
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.remove_signal_handler(sig)
                previous = previous_handlers.get(sig)
                if previous is not None:
                    signal.signal(sig, previous)
            except (NotImplementedError, RuntimeError, ValueError):
                pass


async def run_workflow_local_async(args: argparse.Namespace) -> None:
    """Local (no-Docker) workflow: validates manifest then runs container/run.py in-process."""
    from .local_runner import run_jobs_locally

    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output)

    conn = get_db_connection()
    try:
        if hasattr(args, "manifest") and args.manifest:
            parsed_manifest = require_parsed_manifest(args.manifest, conn)
            pending_jobs = parsed_manifest.plan
        else:
            pending_jobs = generate_execution_plan(conn)
    except ManifestValidationError as exc:
        conn.close()
        console.print("[bold red]Manifest validation failed:[/]")
        for err in exc.parsed.errors:
            console.print(f"  [red]-[/] {err['field']}: {err['message']} ({err['code']})")
        raise SystemExit(2) from exc

    validated_jobs, pre_flight_warnings = validate_pending_jobs(pending_jobs, record_failures=True)
    runnable_jobs = [j for j in validated_jobs if not j.get("_skipped")]
    conn.close()

    if not runnable_jobs:
        logger.info("All jobs were skipped after pre-flight validation.")
        _print_run_summary(validated_jobs, {}, pre_flight_warnings)
        return

    tasks_status = {
        f"{j.get('dataset_name')}_{j.get('model_slug')}": "Pending"
        for j in runnable_jobs
    }

    def update_status(name: str, status: str) -> None:
        tasks_status[name] = status

    console.print(f"[bold]Local run:[/bold] {len(runnable_jobs)} job(s), no Docker")
    with Live(generate_status_table(tasks_status), refresh_per_second=4) as live:
        def _cb(name: str, status: str) -> None:
            update_status(name, status)
            live.update(generate_status_table(tasks_status))

        result_summary = await run_jobs_locally(runnable_jobs, args.seed, status_callback=_cb)

    _print_run_summary(validated_jobs, result_summary, pre_flight_warnings)


def execute_run(args: argparse.Namespace):
    """Executes the model running logic — always uses async/parallel for registry jobs."""
    if getattr(args, "local", False):
        try:
            asyncio.run(run_workflow_local_async(args))
        except KeyboardInterrupt:
            logger.warning("Local workflow interrupted by user")
            raise SystemExit(130)
        return

    conn = get_db_connection()
    try:
        if hasattr(args, "manifest") and args.manifest:
            parsed_manifest = require_parsed_manifest(args.manifest, conn)
            pending_jobs = parsed_manifest.plan
        else:
            pending_jobs = generate_execution_plan(conn)
    except ManifestValidationError as exc:
        conn.close()
        console.print("[bold red]Manifest validation failed:[/]")
        for err in exc.parsed.errors:
            console.print(f"  [red]-[/] {err['field']}: {err['message']} ({err['code']})")
        raise SystemExit(2) from exc
    else:
        conn.close()

    if pending_jobs:
        try:
            asyncio.run(run_workflow_async(args))
        except KeyboardInterrupt:
            logger.warning("Workflow interrupted by user")
            raise SystemExit(130)

        if args.evaluate:
            logger.info("Starting evaluation of results.")
            try:
                run_evaluation_container(args.input, args.output)
                logger.info("Evaluation finished successfully.")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
        return

    # Legacy mode: no pending jobs in DB, fall back to explicit --models + --input
    if not args.models or not args.input:
        logger.info("No pending jobs in registry and no models/input provided for legacy mode.")
        return

    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output)
    logger.info(f"Running models: {args.models}")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")

    for model in args.models:
        logger.info(f"Running model: {model}")
        try:
            run_model_container(model, args.input, args.output, seed=args.seed)
            logger.info(f"Model {model} finished successfully.")
        except Exception as e:
            logger.error(f"Error running model {model}: {e}")

    if args.evaluate:
        logger.info("Starting evaluation of results.")
        try:
            run_evaluation_container(args.input, args.output)
            logger.info("Evaluation finished successfully.")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")


def main():
    """Main entry point for the multiverse CLI."""
    parser = argparse.ArgumentParser(description="Multiverse CLI")

    def add_run_args(p):
        p.add_argument(
            "--models",
            nargs="+",
            required=False,
            help="List of models to run (legacy mode; e.g., pca mofa multivi totalvi)",
            default=[],
        )
        p.add_argument("--input", required=False, help="Path to the input data directory")
        p.add_argument("--output", required=True, help="Path to the output results directory")
        p.add_argument("--seed", type=int, default=42, help="Random seed")
        p.add_argument(
            "--evaluate",
            required=False,
            action="store_true",
            help="Whether to run evaluation after model execution",
        )
        p.add_argument(
            "--concurrent",
            action="store_true",
            help="Deprecated flag kept for backward compatibility; registry runs are always parallel",
        )
        p.add_argument(
            "--manifest",
            required=False,
            help="Path to a run_manifest.yaml file defining jobs to execute",
        )
        p.add_argument(
            "--local",
            action="store_true",
            default=False,
            help=(
                "Run models locally without Docker by calling container/run.py directly. "
                "Requires model Python deps installed on the host. "
                "Use with --manifest for manifest-driven local dev runs."
            ),
        )

    import sys
    known_commands = ["run", "register-dataset", "preprocess-dataset", "register-model", "init-db", "models"]

    if len(sys.argv) > 1 and sys.argv[1] not in known_commands and not sys.argv[1].startswith("-h"):
        add_run_args(parser)
        args = parser.parse_args()
        execute_run(args)
        return

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run models")
    add_run_args(run_parser)

    reg_parser = subparsers.add_parser("register-dataset", help="Register a dataset from dataset.yaml")
    reg_parser.add_argument("--slug", required=False, help="Dataset slug under store/datasets/<slug>/")
    reg_parser.add_argument("--manifest", required=False, help="Explicit path to dataset.yaml")
    reg_parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing registry row when manifest changed.",
    )

    preproc_parser = subparsers.add_parser(
        "preprocess-dataset",
        help="Fuse raw modality files into processed.h5mu (run after register-dataset)",
    )
    preproc_parser.add_argument("--slug", required=False, help="Dataset slug under store/datasets/<slug>/")
    preproc_parser.add_argument("--manifest", required=False, help="Explicit path to dataset.yaml")

    subparsers.add_parser("init-db", help="Initialize the registry database")

    model_reg_parser = subparsers.add_parser(
        "register-model", help="Register a model from model.yaml"
    )
    model_reg_parser.add_argument("--slug", required=False, help="Model slug under store/models/<slug>/")
    model_reg_parser.add_argument("--manifest", required=False, help="Explicit path to model.yaml")
    model_reg_parser.add_argument(
        "--build",
        action="store_true",
        help="Build local Docker image after registration",
    )

    models_parser = subparsers.add_parser("models", help="Model registry/build commands")
    models_subparsers = models_parser.add_subparsers(dest="models_command", help="Models commands")

    models_register = models_subparsers.add_parser(
        "register", help="Register a model from model.yaml"
    )
    models_register.add_argument("--slug", required=False, help="Model slug under store/models/<slug>/")
    models_register.add_argument("--manifest", required=False, help="Explicit path to model.yaml")
    models_register.add_argument(
        "--build",
        action="store_true",
        help="Build local Docker image after registration",
    )

    models_build = models_subparsers.add_parser(
        "build", help="Build a model image locally from model.yaml"
    )
    models_build.add_argument("--slug", required=False, help="Model slug under store/models/<slug>/")
    models_build.add_argument("--manifest", required=False, help="Explicit path to model.yaml")

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("Database and directories initialized.")
    elif args.command == "register-dataset":
        init_db()
        manifest_path = resolve_manifest_path(manifest_path=args.manifest, slug=args.slug)
        try:
            result = register_from_manifest(str(manifest_path), update=True if args.update else None)
        except RuntimeError as exc:
            prompt = f"{exc} Update now? [y/N]: "
            choice = input(prompt).strip().lower()
            if choice in {"y", "yes"}:
                result = register_from_manifest(str(manifest_path), update=True)
            else:
                print("Skipped update.")
                return
        print(result["message"])
        print(f"Dataset slug '{result['slug']}' registered with ID: {result['dataset_id']}")
    elif args.command == "preprocess-dataset":
        manifest_path = resolve_manifest_path(manifest_path=args.manifest, slug=args.slug)
        output = preprocess_dataset(str(manifest_path))
        print(f"Processed dataset written to: {output}")
    elif args.command == "run":
        execute_run(args)
    elif args.command == "register-model":
        model_manifest = resolve_model_manifest_path(
            manifest_path=args.manifest, slug=args.slug
        )
        result = register_model_from_manifest(
            str(model_manifest),
            build=args.build,
        )
        print(result["message"])
        print(f"Model slug '{result['slug']}' registered at version {result['version']}.")
    elif args.command == "models":
        if args.models_command == "register":
            model_manifest = resolve_model_manifest_path(
                manifest_path=args.manifest, slug=args.slug
            )
            result = register_model_from_manifest(
                str(model_manifest),
                build=args.build,
            )
            print(result["message"])
            if args.build:
                print(f"Built local image: {result['docker_image']}")
        elif args.models_command == "build":
            model_manifest = resolve_model_manifest_path(
                manifest_path=args.manifest, slug=args.slug
            )
            manifest = load_model_manifest(str(model_manifest))
            built = build_local_model(manifest)
            if built:
                print(f"Built local image: {built}")
            else:
                print("Remote image expected, skipping local build.")
        else:
            models_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
