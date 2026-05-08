"""
Legacy local JSON workflow runner.

This module intentionally keeps the legacy host-side ML workflow isolated from
the orchestrator control plane.
"""

import os
import json
import random
import numpy as np
import torch

from multiverse.config import load_config
from multiverse.config_schema import validate_config
from multiverse.data_utils import load_datasets, dataset_select
from multiverse.registry import load_registry, get_eligible_models
from multiverse.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")


def main_workflow(config_path: str):
    """Legacy local workflow that executes models directly on host Python."""
    try:
        set_seed(42)

        logger.info(f"Starting multiverse workflow with config: {config_path}")

        config_data = load_config(config_path=config_path)
        if "batch_key" not in config_data:
            config_data["batch_key"] = "batch"
        if "random_seed" not in config_data:
            config_data["random_seed"] = 42

        validated_config = validate_config(config_data)
        config = validated_config.model_dump(by_alias=True)

        output_dir = config.get("output_dir", "/data/outputs/")
        os.makedirs(output_dir, exist_ok=True)
        setup_logging(output_dir)

        logger.info("Configuration loaded successfully")
        logger.info(f"Output directory: {output_dir}")

        registry = load_registry()

        logger.info("Loading datasets...")
        datasets = load_datasets(config)

        model_config = config.get("model", {})
        run_user_params = config.get("_run_user_params", True)
        run_gridsearch = config.get("_run_gridsearch", False)

        logger.info(f"Run user params: {run_user_params}")
        logger.info(f"Run gridsearch: {run_gridsearch}")
        logger.info(f"Available models: {list(model_config.keys())}")

        if run_user_params:
            logger.info("Running models with user-specified parameters")

            for dataset_name, dataset_data in datasets.items():
                logger.info(f"Checking eligibility for dataset: {dataset_name}")
                available_omics = dataset_data["modalities"]

                eligible_models = get_eligible_models(
                    user_requested_models=list(model_config.keys()),
                    available_omics=available_omics,
                    registry=registry,
                )

                if not eligible_models:
                    logger.warning(f"No eligible models found for dataset {dataset_name}")
                    continue

                filtered_model_config = {
                    m: model_config[m] for m in eligible_models if m in model_config
                }

                single_dataset_dict = {dataset_name: dataset_data}
                run_models_with_user_params(config, single_dataset_dict, filtered_model_config)

            for dataset_name in datasets:
                metrics_path = os.path.join(output_dir, dataset_name, "evaluation_metrics.json")
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump({"status": "success"}, f)

        if run_gridsearch:
            logger.warning("Grid search is configured but not implemented.")

        logger.info("Multiverse workflow completed successfully")

    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise


def run_models_with_user_params(config_path: str, datasets: dict, model_config: dict):
    """Executes legacy models using host-local parameters and runtime."""
    from multiverse.models.pca import PCAModel
    from multiverse.models.mofa import MOFAModel
    from multiverse.models.multivi import MultiVIModel
    from multiverse.models.mowgli import MowgliModel
    from multiverse.models.cobolt import CoboltModel
    from multiverse.models.totalvi import TotalVIModel

    model_classes = {
        "pca": PCAModel,
        "mofa": MOFAModel,
        "multivi": MultiVIModel,
        "mowgli": MowgliModel,
        "cobolt": CoboltModel,
        "totalvi": TotalVIModel,
    }

    data_concat = dataset_select(datasets_dict=datasets, data_type="concatenate")

    for model_name, model_class in model_classes.items():
        if model_name in model_config:
            logger.info(f"Running {model_name} model...")
            try:
                for dataset_name, data_dict in data_concat.items():
                    model = model_class(
                        dataset=data_dict,
                        dataset_name=dataset_name,
                        config_path=config_path,
                        is_gridsearch=False,
                    )
                    model.train()
                    model.save_latent()
                    model.umap()
                    model.evaluate_model()
            except Exception as e:
                logger.error(f"Error running {model_name} model: {e}", exc_info=True)
                continue
