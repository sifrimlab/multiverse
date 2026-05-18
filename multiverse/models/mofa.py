import json
import random
from typing import Union

import anndata as ad
import muon as mu
import numpy as np
from ..logging_utils import get_logger
from .runtime_io import (
    build_model_config,
    load_input_mudata,
    load_job_spec,
    setup_container_logging,
)

from .base import ModelFactory

logger = get_logger(__name__)

class MOFAModel(ModelFactory):
    """MOFA model wrapper using the `muon` library.

    Attributes:
        device (str): Computation device (e.g., "cpu", "cuda:0").
        n_iterations (int): Maximum number of iterations for training.
        n_factors (int): Number of latent factors to extract.
        gpu_mode (bool): Flag indicating if GPU acceleration is used.
    """

    def __init__(
        self,
        dataset: ad.AnnData,
        dataset_name: str,
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
    ):
        """Initializes the MOFAModel.

        Args:
            dataset (ad.AnnData): The input dataset (MuData-derived AnnData).
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'mofa' configuration is not found in the model parameters.
        """
        logger.info("Initializing MOFA Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="mofa", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        mofa_params= self.model_params.get(self.model_name)

        self.device = mofa_params.get("device")
        self.n_iterations = mofa_params.get("n_iterations")
        self.umap_color_type = mofa_params.get("umap_color_type")
        self.n_factors = mofa_params.get("n_factors")
        self.umap_random_state = mofa_params.get("umap_random_state")
        self.gpu_mode = self.device != "cpu"

    def train(self):
        """Trains the MOFA+ model using variational inference."""
        logger.info("Training MOFA+ Model")
        try:
            mu.tl.mofa(
                data=self.dataset, n_factors=self.n_factors, gpu_mode=self.gpu_mode
            )
            self.dataset.obsm[self.latent_key] = self.dataset.obsm["X_mofa"]
            logger.info("MOFA training completed.")

            # Compute explained variance if not available
            if "explained_variance" in self.dataset.uns.get("mofa", {}):
                self.explained_variance = self.dataset.uns["mofa"]["explained_variance"]
                logger.info(f"Explained variance per factor: {self.explained_variance}")
            else:
                # Manually calculate explained variance
                self.explained_variance = self._compute_explained_variance()
                logger.info(
                    f"Computed explained variance per factor: {self.explained_variance}"
                )

            logger.info(f"Total explained variance: {sum(self.explained_variance)}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _compute_explained_variance(self):
        """Computes the variance explained by each latent factor.

        Returns:
            np.ndarray: An array containing the explained variance ratio for each factor.
        """
        try:
            factors = self.dataset.obsm[self.latent_key]

            # Total variance from raw data across modalities (dense where needed).
            total_variance = 0
            for modality in self.dataset.mod.values():
                if hasattr(modality.X, "toarray"):
                    modality_data = modality.X.toarray()
                else:
                    modality_data = modality.X
                total_variance += np.var(modality_data, axis=0).sum()

            factor_variances = np.var(factors, axis=0)
            explained_variance_ratio = factor_variances / total_variance
            return explained_variance_ratio

        except Exception as e:
            logger.error(f"Error computing explained variance: {e}")
            return []

    def evaluate_model(self):
        """Evaluates the MOFA+ model by calculating total explained variance.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        if hasattr(self, "explained_variance"):
            if requested is None or "total_variance" in requested:
                total_variance = sum(self.explained_variance)
                logger.info(f"Total Explained Variance (MOFA+): {total_variance}")
                metrics["total_variance"] = total_variance
        else:
            logger.warning("Explained variance not available for MOFA+.")

        self.write_metrics(metrics)


def main():
    setup_container_logging()
    job_spec = load_job_spec()
    config = build_model_config(model_name="mofa", job_spec=job_spec)
    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    data_concat = load_input_mudata()
    dataset_name = job_spec.get("dataset_name", "dataset")

    try:
        model = MOFAModel(
            dataset=data_concat,
            dataset_name=dataset_name,
            config_path=config,
        )
        logger.info(f"Running MOFA model on dataset: {dataset_name}")
        model.train()
        model.save_latent()
        model.umap()
        model.evaluate_model()
        logger.info(f"MOFA model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during MOFA model run: {e}")
        raise

if __name__ == "__main__":
    main()
