import json
import random
from typing import Union

import anndata as ad
import numpy as np
import scvi
from ..data_utils import anndata_concatenate
from ..logging_utils import get_logger
from ..utils import get_device
from .base import ModelFactory
from .runtime_io import (
    build_model_config,
    load_input_mudata,
    load_job_spec,
    setup_container_logging,
)

logger = get_logger(__name__)

class TotalVIModel(ModelFactory):
    """TotalVI model wrapper from the `scvi-tools` library.

    Designed for joint analysis of single-cell RNA and protein data.

    Attributes:
        max_epochs (int): Maximum number of training epochs.
        learning_rate (float): Learning rate for training.
        latent_dimensions (int): Dimension of the latent space.
        torch_device (torch.device): Computation device.
    """

    def __init__(
        self,
        dataset: ad.AnnData,
        dataset_name: str,
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
    ):
        """Initializes the TotalVIModel.

        Args:
            dataset (ad.AnnData): Concatenated RNA and Protein AnnData object.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'totalvi' configuration is missing.
        """
        logger.info("Initializing TotalVI Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="totalvi", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        totalvi_params = self.model_params.get(self.model_name)
        self.device = totalvi_params.get("device")
        self.max_epochs = totalvi_params.get("max_epochs")
        self.learning_rate = totalvi_params.get("learning_rate")
        self.umap_color_type = totalvi_params.get("umap_color_type")
        self.torch_device = "cpu"
        self.latent_dimensions = totalvi_params.get("latent_dimensions")
        self.umap_random_state = totalvi_params.get("umap_random_state")
        self.torch_device = get_device(self.device)

        scvi.model.TOTALVI.setup_anndata(
            self.dataset,
            protein_expression_obsm_key="protein_expression",
            batch_key="batch"
        )

        self.model = scvi.model.TOTALVI(self.dataset)

    def train(self):
        """Trains the TotalVI model using variational inference."""
        logger.info("Training TotalVI Model")
        try:
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            logger.info("TotalVI training completed.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        """Evaluates the TotalVI model.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        if hasattr(self.model, "history") and self.model.history:
            if requested is None or "elbo_train" in requested:
                if "elbo_train" in self.model.history:
                    metrics["elbo_train"] = float(
                        self.model.history["elbo_train"].iloc[-1]
                    )
            if requested is None or "reconstruction_loss_train" in requested:
                if "reconstruction_loss_train" in self.model.history:
                    metrics["reconstruction_loss_train"] = float(
                        self.model.history["reconstruction_loss_train"].iloc[-1]
                    )
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
    config = build_model_config(model_name="totalvi", job_spec=job_spec)
    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed
    mudata_obj = load_input_mudata()
    dataset_name = job_spec.get("dataset_name", "dataset")
    data_concat = anndata_concatenate(
        list_anndata=[mudata_obj[modality] for modality in mudata_obj.mod.keys()],
        list_modality=list(mudata_obj.mod.keys()),
    )

    try:
        model = TotalVIModel(
            dataset=data_concat,
            dataset_name=dataset_name,
            config_path=config,
        )
        logger.info(f"Running TotalVI model on dataset: {dataset_name}")
        model.train()
        model.save_latent()
        model.umap()
        model.evaluate_model()
        logger.info(f"TotalVI model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during TotalVI model run: {e}")
        raise

if __name__ == "__main__":
    main()
