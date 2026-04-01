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
    run_models_concurrently
)
from ..logging_utils import get_logger, setup_logging
from ..registry import load_registry
from ..ingestion import register_dataset
from ..registry_db import init_db

logger = get_logger(__name__)

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

    registry = load_registry()
    models_info = []
    for m in args.models:
        if m in registry:
            models_info.append({"name": m, "image": registry[m].docker_image})
        else:
            logger.warning(f"Model {m} not found in registry. Skipping.")

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
        summary = await run_models_concurrently(
            models_info,
            args.input,
            args.seed,
            args.output,
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
        p.add_argument("--input", required=True, help="Path to the input data directory")
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
    reg_parser = subparsers.add_parser("register-dataset", help="Register a dataset")
    reg_parser.add_argument("--path", required=True, help="Path to the dataset file")
    reg_parser.add_argument("--name", required=True, help="Name of the dataset")
    reg_parser.add_argument("--batch-key", required=True, help="Batch key in the dataset")

    # Init DB command
    subparsers.add_parser("init-db", help="Initialize the registry database")

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("Database and directories initialized.")
    elif args.command == "register-dataset":
        init_db()  # Ensure DB is ready
        dataset_id = register_dataset(args.path, args.name, args.batch_key)
        print(f"Dataset '{args.name}' registered with ID: {dataset_id}")
    elif args.command == "run":
        execute_run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
