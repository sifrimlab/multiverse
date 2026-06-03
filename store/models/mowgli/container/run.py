"""Mowgli container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""

import json
import os
import random
from typing import Union

import anndata as ad
import mowgli
import numpy as np
import scanpy as sc
import torch
from mvr_worker import (OUTPUT_DIR, ModelFactory, build_model_config,
                        get_device, get_logger, load_input_mudata,
                        load_job_spec, preprocess_mudata, replay_history,
                        resolve_preprocess_params, setup_container_logging)

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

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="mowgli",
            is_gridsearch=is_gridsearch,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        mowgli_params = self.model_params.get(self.model_name)

        self.device = mowgli_params.get("device")
        self.torch_device = "cpu"
        self.latent_dimensions = mowgli_params.get("latent_dimensions", 20)
        self.optimizer = mowgli_params.get("optimizer", "adam")
        self.learning_rate = mowgli_params.get("learning_rate", 0.001)
        self.inner_tolerance = mowgli_params.get("tol_inner", 1e-6)
        self.max_inner_iteration = mowgli_params.get("max_iter_inner", 500)
        self.umap_color_type = mowgli_params.get("umap_color_type")
        self.umap_random_state = mowgli_params.get("umap_random_state")
        self.loss = 0
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        self.torch_device = get_device(self.device)
        self.dataset_name = dataset_name
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
                max_iter_inner=self.max_inner_iteration,
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
        history: dict = {}
        if hasattr(self, "loss"):
            if requested is None or "ot_loss" in requested:
                ot_loss = float(-self.loss)
                logger.info(f"Optimal Transport Loss (Mowgli): {ot_loss}")
                metrics["ot_loss"] = ot_loss
        else:
            logger.warning("Loss not available in the model.")

        if hasattr(self.model, "losses") and self.model.losses:
            history["ot_loss"] = [float(-value) for value in self.model.losses]
            history = replay_history(
                history,
                output_dir=OUTPUT_DIR,
                run_name=f"{self.dataset_name}-mowgli-{os.path.basename(OUTPUT_DIR)}",
            )

        self.write_metrics(metrics, history=history or None)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("mowgli", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_name = job_spec.get("dataset_slug", "dataset")
    try:
        mdata = load_input_mudata()
        # TODO: make cell type and batch key not hardcoded
        modalities = list(mdata.mod.keys())
        # Preprocessing is resolved from the job spec (run manifest / GUI),
        # falling back to these built-in defaults when unspecified (issue #22).
        config["preprocess_params"] = resolve_preprocess_params(
            job_spec,
            modalities,
            {
                "n_top_genes": 1000,
                "scale": {
                    mod: True for mod in modalities
                },  # Scale all modalities by default
                "normalization_target_sum": 1e4,
                "log_normalization": True,
            },
        )
        mdata = preprocess_mudata(
            mdata,
            config["preprocess_params"],
            cell_type_key="cell_type",
            batch_key="batch",
        )
    except Exception as e:
        logger.error(f"Failed to load and concatenate input data: {e}")
        raise

    try:
        model = MowgliModel(
            dataset=mdata,
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
