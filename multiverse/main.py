"""
Main workflow execution module for multiverse.

This module provides the main_workflow function that orchestrates the execution
of multimodal data integration models based on a configuration file.
"""

import os
import sys
import random
import numpy as np
import torch
from .config import load_config
from .data_utils import load_datasets, dataset_select
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")


def main_workflow(config_path: str):
    """
    Main workflow for running multiverse models.
    
    This function loads the configuration, prepares datasets, and executes
    the specified models based on the configuration settings.
    
    Args:
        config_path (str): Path to the JSON configuration file
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
        Exception: For any other errors during execution
    """
    try:
        # Set random seed for reproducibility at the very start
        set_seed(42)
        
        logger.info(f"Starting multiverse workflow with config: {config_path}")
        
        # Load configuration
        config = load_config(config_path=config_path)
        
        # Setup output directory and logging
        output_dir = config.get("output_dir", "/data/outputs/")
        os.makedirs(output_dir, exist_ok=True)
        setup_logging(output_dir)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Output directory: {output_dir}")
        
        # Load datasets
        logger.info("Loading datasets...")
        datasets = load_datasets(config_path)
        
        # Get model configuration
        model_config = config.get("model", {})
        run_user_params = config.get("_run_user_params", True)
        run_gridsearch = config.get("_run_gridsearch", False)
        
        logger.info(f"Run user params: {run_user_params}")
        logger.info(f"Run gridsearch: {run_gridsearch}")
        logger.info(f"Available models: {list(model_config.keys())}")
        
        # Process each model
        if run_user_params:
            logger.info("Running models with user-specified parameters")
            run_models_with_user_params(config_path, datasets, model_config)
        
        if run_gridsearch:
            logger.warning("Grid search is configured but not yet implemented in the workflow")
            logger.info("Skipping grid search for now")
            # TODO: Implement grid search workflow
            # run_models_with_gridsearch(config_path, datasets, model_config)
        
        logger.info("Multiverse workflow completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise


def run_models_with_user_params(config_path, datasets, model_config):
    """
    Run models with user-specified parameters.
    
    Args:
        config_path (str): Path to the configuration file
        datasets (dict): Dictionary of loaded datasets
        model_config (dict): Model configuration dictionary
    """
    from .models.pca import PCAModel
    from .models.mofa import MOFAModel
    from .models.multivi import MultiVIModel
    from .models.mowgli import MowgliModel
    from .models.cobolt import CoboltModel
    from .models.totalvi import TotalVIModel
    
    # Map model names to their classes
    model_classes = {
        "pca": PCAModel,
        "mofa": MOFAModel,
        "multivi": MultiVIModel,
        "mowgli": MowgliModel,
        "cobolt": CoboltModel,
        "totalvi": TotalVIModel,
    }
    
    # Get concatenated datasets
    data_concat = dataset_select(datasets_dict=datasets, data_type="concatenate")
    
    # Run each model
    for model_name, model_class in model_classes.items():
        if model_name in model_config:
            logger.info(f"Running {model_name} model...")
            try:
                for dataset_name, data_dict in data_concat.items():
                    logger.info(f"Processing dataset: {dataset_name} with {model_name}")
                    
                    # Instantiate the model
                    model = model_class(
                        dataset=data_dict,
                        dataset_name=dataset_name,
                        config_path=config_path,
                        is_gridsearch=False,
                    )
                    
                    # Run the model pipeline
                    model.train()
                    model.save_latent()
                    model.umap()
                    model.evaluate_model()
                    
                    logger.info(f"{model_name} completed for {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Error running {model_name} model: {e}", exc_info=True)
                # Continue with other models to allow partial execution
                continue

