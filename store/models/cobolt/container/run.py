"""Cobolt container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""

import os
import random
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
from cobolt.model import Cobolt
from cobolt.utils import MultiomicDataset, SingleData
from mvr_worker import (OUTPUT_DIR, ModelFactory, anndata_concatenate,
                        build_model_config, get_device, get_logger,
                        load_input_mudata, load_job_spec, preprocess_mudata,
                        replay_history, resolve_labels_key_params,
                        resolve_preprocess_params, series_to_float_list,
                        setup_container_logging)

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
        cell_type_key: str = "cell_type",
        batch_key: str = "batch",
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
            cell_type_key=cell_type_key,
            batch_key=batch_key,
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
        self.dataset_name = dataset_name
        self.modalities = dataset["modalities"]
        # initialize dataset
        self.single_data_list = []
        for modality, adata in zip(self.dataset["modalities"], self.dataset["data"]):
            self.single_data_list.append(
                SingleData(
                    feature_name=modality,
                    dataset_name=self.dataset_name,
                    feature=adata.var_names.to_numpy(),
                    count=sp.csr_matrix(adata.X),
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
            adata_list=self.dataset["data"],
            selected_modalities=self.modalities,
            obs=self.dataset["obs"],
            cell_type_key=cell_type_key,
            batch_key=batch_key,
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
        if (
            hasattr(self.model, "history")
            and self.model.history
            and "loss" in self.model.history
        ):
            loss_series = series_to_float_list(self.model.history["loss"])
            if loss_series:
                history["loss"] = loss_series

        history = replay_history(
            history,
            output_dir=OUTPUT_DIR,
            run_name=f"{self.dataset_name}-cobolt-{os.path.basename(OUTPUT_DIR)}",
        )
        self.write_metrics(metrics, history=history or None)


def main() -> None:
    """Container entry: load job spec and data, preprocess, train, write outputs.

    ``cell_type_key`` and ``batch_key`` are fixed to ``cell_type`` / ``batch`` until
    job-spec overrides exist.
    """
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("cobolt", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mudata_obj = load_input_mudata()
    modalities = list(mudata_obj.mod.keys())
    label_keys = resolve_labels_key_params(job_spec)
    cell_type_key = label_keys["cell_type_key"]
    batch_key = label_keys["batch_key"]
    config["preprocess_params"] = resolve_preprocess_params(
        job_spec,
        modalities,
        {
            "n_top_genes": 1000,
            "scale": {modality: False for modality in modalities},
            "normalization_target_sum": None,
            "log_normalization": False,
        },
    )
    mudata_obj = preprocess_mudata(
        mudata_obj,
        config["preprocess_params"],
        cell_type_key=cell_type_key,
        batch_key=batch_key,
    )

    dataset_name = job_spec.get("dataset_slug", "dataset")
    datasets = {
        dataset_name: {
            "modalities": modalities,
            "data": [mudata_obj[modality] for modality in modalities],
            "obs": mudata_obj.obs,
        }
    }

    try:
        for dataset_name, data_dict in datasets.items():
            model = CoboltModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=config,
                cell_type_key=cell_type_key,
                batch_key=batch_key,
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
