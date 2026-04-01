import argparse
import os
import asyncio
from typing import Dict, List
from rich.live import Live
from rich.table import Table

from .docker_runner import (
    run_model_container,
    run_evaluation_container,
    build_images_concurrently,
    run_models_concurrently,
    run_jobs_concurrently
)
from ..logging_utils import get_logger, setup_logging
from ..registry import load_registry
from ..ingestion import register_from_manifest, resolve_manifest_path
from ..registry_db import init_db, get_db_connection, ARTIFACTS_DIR
import json

logger = get_logger(__name__)

def generate_execution_plan(conn):
    """Generates a list of required ML jobs by checking the database.

    Args:
        conn: A SQLite connection.

    Returns:
        List[Dict]: A list of pending jobs, each containing dataset and model info.
    """
    cursor = conn.cursor()

    # 1. Get all READY datasets
    cursor.execute("SELECT id, name, path, omics_available FROM datasets WHERE status = 'READY'")
    datasets = cursor.fetchall()

    # 2. Get all models
    cursor.execute("SELECT name, docker_image, supported_omics FROM models")
    models = cursor.fetchall()

    pending_jobs = []

    for d_id, d_name, d_path, d_omics_json in datasets:
        d_omics = set(json.loads(d_omics_json))

        for m_name, m_image, m_omics_json in models:
            m_omics = set(json.loads(m_omics_json))

            # Check if model is compatible with dataset omics
            if m_omics.issubset(d_omics):
                # Check if this combination already has a SUCCESS run
                cursor.execute(
                    "SELECT status FROM runs WHERE dataset_id = ? AND model_name = ?",
                    (d_id, m_name)
                )
                run = cursor.fetchone()

                if run is None or run[0] != 'SUCCESS':
                    output_path = os.path.join(ARTIFACTS_DIR, f"{d_name}_{m_name}")
                    pending_jobs.append({
                        "dataset_id": d_id,
                        "dataset_name": d_name,
                        "dataset_path": d_path,
                        "model_name": m_name,
                        "model_image": m_image,
                        "output_path": output_path
                    })

    return pending_jobs

def generate_status_table(tasks: Dict[str, str]) -> Table:
    """Generates a Rich Table representing the current status of all tasks.

    Args:
        tasks (Dict[str, str]): A mapping from task names to their status strings.

    Returns:
        rich.table.Table: A formatted table for terminal display.
    """
    table = Table(title="Multiverse Parallel Execution Dashboard")
    table.add_column("Task/Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", style="magenta")

    for name, status in tasks.items():
        style = "green" if status in ["Success", "Ready"] else "red" if "Failed" in status or "Error" in status else "yellow"
        table.add_row(name, f"[{style}]{status}[/]")

    return table

async def run_workflow_async(args: argparse.Namespace):
    """Executes the concurrent Docker-based workflow.

    Manages image preparation and parallel model execution with a live
    terminal dashboard.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output)

    conn = get_db_connection()
    pending_jobs = generate_execution_plan(conn)
    conn.close()

    if not pending_jobs:
        logger.info("No pending jobs to execute.")
        return

    models_info = []
    for job in pending_jobs:
        models_info.append({
            "name": f"{job['dataset_name']}_{job['model_name']}",
            "image": job['model_image'],
            "dataset_path": job['dataset_path'],
            "output_path": job['output_path'],
            "dataset_id": job['dataset_id'],
            "model_name_orig": job['model_name']
        })

    tasks_status = {m["name"]: "Pending" for m in models_info}
    image_tags = list(set(m["image"] for m in models_info))
    for tag in image_tags:
        tasks_status[tag] = "Queued"

    with Live(generate_status_table(tasks_status), refresh_per_second=4) as live:
        def update_status(name, status):
            tasks_status[name] = status
            live.update(generate_status_table(tasks_status))

        # 1. Build/Pull images
        update_status("Image Preparation", "In Progress")
        try:
            await build_images_concurrently(image_tags, status_callback=update_status)
            update_status("Image Preparation", "Ready")
        except Exception as e:
            update_status("Image Preparation", f"Failed: {e}")
            return

        # 2. Run models
        update_status("Model Execution", "In Progress")

        # We need to adapt run_models_concurrently to handle multiple datasets if needed,
        # but for now we'll run the pending jobs.
        # Note: the current run_models_concurrently assumes a single data_path and output_dir.
        # We need to modify it or call it per dataset.

        # Let's refine run_models_concurrently in docker_runner.py to take the full job info.

        summary = await run_jobs_concurrently(
            models_info,
            args.seed,
            status_callback=update_status
        )

        failed_models = [name for name, status in summary.items() if status == "failed"]
        if failed_models:
             update_status("Model Execution", f"Completed with failures: {failed_models}")
        else:
             update_status("Model Execution", "Success")

def execute_run(args: argparse.Namespace):
    """Executes the model running logic, either sequentially or concurrently."""
    if args.concurrent:
        asyncio.run(run_workflow_async(args))
    else:
        # Check if we should use DB-based planner or legacy mode
        conn = get_db_connection()
        pending_jobs = generate_execution_plan(conn)
        conn.close()

        if pending_jobs:
            logger.info(f"Running {len(pending_jobs)} pending jobs from registry.")
            for job in pending_jobs:
                logger.info(f"Running model {job['model_name']} on dataset {job['dataset_name']}")
                try:
                    run_model_container(
                        job['model_name'],
                        job['dataset_path'],
                        job['output_path']
                    )
                    # Record success in DB
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO runs (dataset_id, model_name, status, output_path) VALUES (?, ?, ?, ?)",
                        (job['dataset_id'], job['model_name'], "SUCCESS", job['output_path'])
                    )
                    conn.commit()
                    conn.close()
                    logger.info(f"Model {job['model_name']} on {job['dataset_name']} finished successfully.")
                except Exception as e:
                    logger.error(f"Error running model {job['model_name']} on {job['dataset_name']}: {e}")
                    # Record failure in DB
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO runs (dataset_id, model_name, status, output_path) VALUES (?, ?, ?, ?)",
                        (job['dataset_id'], job['model_name'], "FAILED", job['output_path'])
                    )
                    conn.commit()
                    conn.close()
            return

        # Legacy mode if no pending jobs in DB (requires --models and --input)
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
                run_model_container(model, args.input, args.output)
                logger.info(f"Model {model} finished successfully.")
            except Exception as e:
                logger.error(f"Error running model {model}: {e}")
                pass

    if args.evaluate:
        logger.info("Starting evaluation of results.")
        try:
            run_evaluation_container(args.input, args.output)
            logger.info("Evaluation finished successfully.")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")

def main():
    """Main entry point for the multiverse CLI.

    Supports both sequential and concurrent Docker-based model execution.
    """
    parser = argparse.ArgumentParser(description="Multiverse CLI")

    # Common arguments for 'run' and legacy mode
    def add_run_args(p):
        p.add_argument(
            "--models",
            nargs="+",
            required=False,
            help="List of models to run (e.g., pca mofa multivi totalvi)",
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
            help="Run models concurrently using Docker",
        )

    # If the first argument is not a known command, we assume legacy mode (run)
    import sys
    known_commands = ["run", "register-dataset", "init-db"]

    if len(sys.argv) > 1 and sys.argv[1] not in known_commands and not sys.argv[1].startswith("-h"):
        # Legacy mode: no command provided
        add_run_args(parser)
        args = parser.parse_args()
        execute_run(args)
        return

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run models")
    add_run_args(run_parser)

    # Register dataset command
    reg_parser = subparsers.add_parser("register-dataset", help="Register a dataset from dataset.yaml")
    reg_parser.add_argument("--slug", required=False, help="Dataset slug under store/datasets/<slug>/")
    reg_parser.add_argument("--manifest", required=False, help="Explicit path to dataset.yaml")
    reg_parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing registry row when manifest changed.",
    )

    # Init DB command
    subparsers.add_parser("init-db", help="Initialize the registry database")

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("Database and directories initialized.")
    elif args.command == "register-dataset":
        init_db()  # Ensure DB is ready
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
    elif args.command == "run":
        execute_run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
