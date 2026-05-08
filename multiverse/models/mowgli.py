import json
import random
from typing import Union

import anndata as ad
import mowgli
import numpy as np
import torch
from .base import ModelFactory
from ..logging_utils import get_logger
from ..utils import get_device
from .runtime_io import (
    build_model_config,
    load_input_mudata,
    load_job_spec,
    setup_container_logging,
)

logger = get_logger(__name__)

class MowgliModel(ModelFactory):
    """Mowgli model wrapper.

    Uses Optimal Transport and Non-negative Matrix Factorization for multimodal
    data integration.

    Attributes:
        latent_dimensions (int): Dimension of the latent space.
        optimizer (str): Name of the optimizer to use.
        learning_rate (float): Learning rate for the optimizer.
        loss (float): Final training loss.
    """

    def __init__(
        self,
        dataset: ad.AnnData,
        dataset_name: str,
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
    ):
        """Initializes the MowgliModel.

        Args:
            dataset (ad.AnnData): The input dataset.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'mowgli' configuration is not found in the model parameters.
        """
        logger.info("Initializing Mowgli Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="mowgli", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        mowgli_params = self.model_params.get(self.model_name)

        self.device = mowgli_params.get("device")
        self.torch_device = 'cpu'
        self.latent_dimensions = mowgli_params.get("latent_dimensions")
        self.optimizer = mowgli_params.get("optimizer")
        self.learning_rate = mowgli_params.get("learning_rate")
        self.inner_tolerance = mowgli_params.get("tol_inner")
        self.max_inner_iteration = mowgli_params.get("max_iter_inner")
        self.umap_color_type = mowgli_params.get("umap_color_type")
        self.umap_random_state = mowgli_params.get("umap_random_state")
        self.loss = 0
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        self.torch_device = get_device(self.device)
        logger.info(f"Mowgli model initiated with {self.latent_dimensions} dimension.")

    def train(self):
        """Trains the Mowgli model using the specified optimizer."""
        logger.info("Training Mowgli Model")
        try:
            self.model.train(
                self.dataset,
                device=self.torch_device,
                optim_name=self.optimizer,
                lr=self.learning_rate,
                tol_inner=self.inner_tolerance,
                max_iter_inner=self.max_inner_iteration
            )
            self.dataset.obsm[self.latent_key] = self.dataset.obsm["W_OT"]
            self.loss = self.model.losses[-1]
            logger.info(f"Final training loss: {self.loss}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        """Evaluates the Mowgli model by reporting the final Optimal Transport loss.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        if hasattr(self, "loss"):
            if requested is None or "ot_loss" in requested:
                logger.info(f"Optimal Transport Loss (Mowgli): {self.loss}")
                metrics["ot_loss"] = str(-self.loss)
        else:
            logger.warning("Loss not available in the model.")

        try:
            with open(self.metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_filepath}")
        except IOError as e:
            logger.error(f"Could not write metrics file to {self.metrics_filepath}: {e}")
            raise


def main():
    setup_container_logging()
    job_spec = load_job_spec()
    config = build_model_config(model_name="mowgli", job_spec=job_spec)
    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_concat = load_input_mudata()
    dataset_name = job_spec.get("dataset_name", "dataset")

    try:
        model = MowgliModel(
            dataset=data_concat,
            dataset_name=dataset_name,
            config_path=config,
        )
        logger.info(f"Running Mowgli model on dataset: {dataset_name}")
        model.train()
        model.save_latent()
        model.umap()
        model.evaluate_model()
        logger.info(f"Mowgli model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during Mowgli model run: {e}")
        raise

if __name__ == "__main__":
    main()
