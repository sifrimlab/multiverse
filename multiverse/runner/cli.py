import argparse
import json
import os
import asyncio
from typing import Dict, List, Tuple
from rich.live import Live
from rich.table import Table
from rich.console import Console

from .docker_runner import (
    run_model_container,
    run_evaluation_container,
    build_images_concurrently,
    run_models_concurrently,
    run_jobs_concurrently,
    run_job_container_sync,
    ensure_image_prepared,
    start_db_writer,
    stop_db_writer,
)
from ..logging_utils import get_logger, setup_logging
from ..ingestion import register_from_manifest, resolve_manifest_path, preprocess_dataset
from ..registry_db import init_db, get_db_connection, ARTIFACTS_DIR
from ..models_ingest import (
    resolve_model_manifest_path,
    load_model_manifest,
    register_model_from_manifest,
)
from ..builder import build_local_model
import yaml

logger = get_logger(__name__)
console = Console()

# Required omics per model slug. Empty set means any combination is acceptable.
MODEL_REQUIRED_OMICS = {
    "multivi": {"rna", "atac"},
    "totalvi": {"rna", "adt"},
    "cobolt": {"rna", "atac"},
    "mofa": set(),
    "mowgli": set(),
    "pca": set(),
}


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


def _read_obs_columns(path: str) -> set:
    """Return the set of obs column names from an h5ad or h5mu file via h5py."""
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            obs = f.get("obs")
            if obs is not None:
                return {k for k in obs.keys() if not k.startswith("_")}
    except Exception as exc:
        logger.debug(f"Could not read obs columns from {path}: {exc}")
    return set()


def _read_batch_count(path: str, batch_key: str) -> int:
    """Return number of unique batch values. 0 if unreadable."""
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            obs = f.get("obs")
            if obs is None or batch_key not in obs:
                return 0
            batch_node = obs[batch_key]
            # Categorical encoding: values are stored under 'codes', categories under 'categories'
            if "categories" in batch_node:
                return int(batch_node["categories"].shape[0])
            # Plain array
            import numpy as np
            return int(len(set(batch_node[:])))
    except Exception as exc:
        logger.debug(f"Could not read batch count from {path}: {exc}")
    return 0


def validate_pending_jobs(
    pending_jobs: List[Dict],
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
    batch_count_cache: Dict[int, int] = {}

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
            logger.warning(
                f"[SKIP] {dataset_name}/{model_slug}: model requires omics {required}, "
                f"dataset has {omics_available}"
            )
            job["_skip_reason"] = f"missing required omics {required - omics_available}"
            validated.append({**job, "_skipped": True})
            continue

        # 2. Batch key presence check (only if registry declares a batch_key)
        if batch_key:
            if dataset_id not in obs_cache:
                obs_cache[dataset_id] = _read_obs_columns(dataset_path)
            obs_cols = obs_cache[dataset_id]
            if obs_cols and batch_key not in obs_cols:
                logger.warning(
                    f"[SKIP] {dataset_name}/{model_slug}: batch_key '{batch_key}' "
                    f"not found in dataset obs columns"
                )
                job["_skip_reason"] = f"batch_key '{batch_key}' absent from dataset obs"
                validated.append({**job, "_skipped": True})
                continue

        # 3. Cell type key warning (don't skip)
        if cell_type_key:
            if dataset_id not in obs_cache:
                obs_cache[dataset_id] = _read_obs_columns(dataset_path)
            obs_cols = obs_cache[dataset_id]
            if obs_cols and cell_type_key not in obs_cols:
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


def load_manifest(manifest_path: str) -> Dict:
    """Loads and parses the run manifest YAML file."""
    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


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

            cursor.execute(
                "SELECT status FROM runs WHERE dataset_id = ? AND model_slug = ? AND model_version = ?",
                (d_id, m_slug, m_version)
            )
            run = cursor.fetchone()

            if run is None or run[0] != 'SUCCESS':
                # Merge global metrics with per-job override
                job_metrics = job.get("metrics", {})
                merged_metrics = {**global_metrics, **job_metrics}

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
                    "model_params": job.get("model_params", {}) or {},
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

    # Start the single-writer DB actor.  All DB writes from parallel worker
    # coroutines are funnelled through this task; the planner and GUI use
    # their own short-lived read connections (WAL mode keeps them non-blocking).
    _db_writer = start_db_writer()

    conn = get_db_connection()
    manifest_data = {}
    if hasattr(args, "manifest") and args.manifest:
        manifest_data = load_manifest(args.manifest)
        pending_jobs = generate_execution_plan_from_manifest(conn, manifest_data)
    else:
        pending_jobs = generate_execution_plan(conn)
    conn.close()

    # ResourcePool capacity — override via manifest globals.host_ram_gb if set
    host_ram_gb: float | None = manifest_data.get("globals", {}).get("host_ram_gb")

    if not pending_jobs:
        logger.info("No pending jobs to execute.")
        return

    # Pre-flight validation gate
    validated_jobs, pre_flight_warnings = validate_pending_jobs(pending_jobs)
    runnable_jobs = [j for j in validated_jobs if not j.get("_skipped")]

    if not runnable_jobs:
        logger.info("All jobs were skipped after pre-flight validation.")
        _print_run_summary(validated_jobs, {}, pre_flight_warnings)
        return

    experiment_name = os.path.basename(os.path.normpath(args.output)) if args.output else "default_experiment"
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
        }
        if job.get("mem_limit"):
            job_entry["mem_limit"] = job["mem_limit"]
        models_info.append(job_entry)

    tasks_status = {m["name"]: "Pending" for m in models_info}
    # Deduplicated image tags from all runnable jobs
    image_tags = list(set(m["image"] for m in models_info))
    for tag in image_tags:
        tasks_status[tag] = "Queued"

    result_summary: Dict[str, str] = {}

    with Live(generate_status_table(tasks_status), refresh_per_second=4) as live:
        def update_status(name, status):
            tasks_status[name] = status
            live.update(generate_status_table(tasks_status))

        # 1. Build/pull ALL required images before any job starts
        update_status("Image Preparation", "In Progress")
        try:
            await build_images_concurrently(image_tags, status_callback=update_status)
            update_status("Image Preparation", "Ready")
        except Exception as e:
            update_status("Image Preparation", f"Failed: {e}")
            _print_run_summary(validated_jobs, {}, pre_flight_warnings)
            return

        # 2. Run all jobs in parallel
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

    # End-of-run summary (outside Live context so it renders cleanly)
    _print_run_summary(validated_jobs, result_summary, pre_flight_warnings)

    # Drain the DB queue and shut down the writer task gracefully.
    # This guarantees all PROMOTING→SUCCESS updates are committed before
    # execute_run returns and before any downstream evaluation step reads
    # the runs table.
    await stop_db_writer()


def execute_run(args: argparse.Namespace):
    """Executes the model running logic — always uses async/parallel for registry jobs."""
    # Registry-based runs (from manifest or DB) always go through the async parallel path.
    conn = get_db_connection()
    if hasattr(args, "manifest") and args.manifest:
        manifest_data = load_manifest(args.manifest)
        pending_jobs = generate_execution_plan_from_manifest(conn, manifest_data)
    else:
        pending_jobs = generate_execution_plan(conn)
    conn.close()

    if pending_jobs:
        asyncio.run(run_workflow_async(args))

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
