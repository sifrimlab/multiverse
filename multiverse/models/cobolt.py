import json
import random
from typing import Union

import numpy as np
import torch
from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt

from .base import ModelFactory
from .metrics_utils import series_to_float_list
from ..data_utils import anndata_concatenate
from ..logging_utils import get_logger
from ..utils import get_device
from .runtime_io import (
    build_model_config,
    load_input_mudata,
    load_job_spec,
    setup_container_logging,
)

logger = get_logger(__name__)


class CoboltModel(ModelFactory):
    """Cobolt model wrapper.

    Integrates multimodal data using a Bayesian hierarchical model.

    Attributes:
        latent_dimensions (int): Dimension of the latent space.
        learning_rate (float): Learning rate for training.
        num_epochs (int): Number of training epochs.
        loss (float): Final training loss.
    """

    def __init__(
        self,
        dataset: dict,
        dataset_name: str,
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
    ):
        """Initializes the CoboltModel.

        Args:
            dataset (dict): Dictionary containing modality names and AnnData objects.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'cobolt' configuration is not found in the model parameters.
        """
        logger.info("Initializing Cobolt Model")

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="cobolt",
            is_gridsearch=is_gridsearch,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        cobolt_params = self.model_params.get(self.model_name)

        self.device = cobolt_params.get("device")
        self.torch_device = "cpu"
        self.latent_dimensions = cobolt_params.get("latent_dimensions")
        self.umap_color_type = cobolt_params.get("umap_color_type")
        self.umap_random_state = cobolt_params.get("umap_random_state")
        self.learning_rate = cobolt_params.get("learning_rate")
        self.num_epochs = cobolt_params.get("num_epochs")
        self.loss = 0
        self.torch_device = get_device(self.device)
        # initialize dataset
        self.single_data_list = []
        for modality, adata in zip(self.dataset["modalities"], self.dataset["data"]):
            self.single_data_list.append(
                SingleData(
                    feature_name=modality,
                    dataset_name=self.dataset_name,
                    feature=adata.var_names.to_numpy(),
                    count=adata.X,
                    barcode=adata.obs_names.to_numpy(),
                )
            )

        self.multiomic_dataset = MultiomicDataset.from_singledata(
            *self.single_data_list
        )

        self.model = Cobolt(
            dataset=self.multiomic_dataset,
            n_latent=self.latent_dimensions,
            lr=self.learning_rate,
            device=self.torch_device,
        )

        logger.info(f"Cobolt model initiated with {self.latent_dimensions} dimension.")

        self.dataset = anndata_concatenate(
            list_anndata=self.dataset["data"], list_modality=self.dataset["modalities"]
        )

    def train(self):
        """Trains the Cobolt model."""
        logger.info("Training Cobolt Model")
        try:
            self.model.train(num_epochs=self.num_epochs)
            self.loss = self.model.history["loss"][-1]  # Get the last loss value
            # Save the latent embeddings for cells present in all modalities (intersection).
            self.dataset.obsm[self.latent_key] = self.model.get_all_latent()[0][
                [
                    self.multiomic_dataset.get_comb_idx(
                        [True] * len(self.multiomic_dataset.omic)
                    )
                ]
            ].squeeze(0)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        """Evaluates the Cobolt model by reporting the final training loss.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        history: dict = {}
        if hasattr(self, "loss"):
            if requested is None or "loss" in requested:
                logger.info(f"Cobolt Loss: {self.loss}")
                metrics["loss"] = float(self.loss)
        else:
            logger.warning("Loss not available in the model.")
        if hasattr(self.model, "history") and self.model.history and "loss" in self.model.history:
            loss_series = series_to_float_list(self.model.history["loss"])
            if loss_series:
                history["loss"] = loss_series
        self.write_metrics(metrics, history=history or None)


def main():
    setup_container_logging()
    job_spec = load_job_spec()
    config = build_model_config(model_name="cobolt", job_spec=job_spec)
    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mudata_obj = load_input_mudata()
    dataset_name = job_spec.get("dataset_name", "dataset")
    modalities = list(mudata_obj.mod.keys())
    datasets = {
        dataset_name: {
            "modalities": modalities,
            "data": [mudata_obj[modality] for modality in modalities],
        }
    }

    try:
        for dataset_name, data_dict in datasets.items():
            model = CoboltModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=config,
            )
            logger.info(f"Running Cobolt model on dataset: {dataset_name}")
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            logger.info(f"Cobolt model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during Cobolt model run: {e}")
        raise


if __name__ == "__main__":
    main()
