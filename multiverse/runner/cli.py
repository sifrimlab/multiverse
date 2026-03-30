import argparse
import os
from .docker_runner import run_model_container, run_evaluation_container
from ..logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run multiverse models in Docker containers")
    parser.add_argument(
        "--models",
        nargs="+",
        required=False,
        help="List of models to run (e.g., pca mofa multivi totalvi)",
        default=[],
    )
    parser.add_argument("--input", required=True, help="Path to the input data directory")
    parser.add_argument("--output", required=True, help="Path to the output results directory")
    parser.add_argument(
        "--evaluate",
        required=False,
        action="store_true",
        help="Whether to run evaluation after model execution",
    )
    args = parser.parse_args()

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
            # Decide if you want to continue with other models or stop
            # For now, we'll just print the error and continue
            pass

    if args.evaluate:
        logger.info("Starting evaluation of results.")
        try:
            run_evaluation_container(args.input, args.output)
            logger.info("Evaluation finished successfully.")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Decide if you want to continue with other models or stop
            # For now, we'll just print the error and continue
            pass
if __name__ == "__main__":
    main()
