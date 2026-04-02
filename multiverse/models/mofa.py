import argparse
import os
import json
from typing import Union

import anndata as ad
import muon as mu
import numpy as np
from ..config import load_config
from ..logging_utils import get_logger, setup_logging
from ..data_utils import load_datasets, dataset_select

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
        metrics = {}
        if hasattr(self, "explained_variance"):
            total_variance = sum(self.explained_variance)
            logger.info(f"Total Explained Variance (MOFA+): {total_variance}")
            metrics["total_variance"] = total_variance
        else:
            logger.warning("Explained variance not available for MOFA+.")

        try:
            with open(self.metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_filepath}")
        except IOError as e:
            logger.error(
                f"Could not write metrics file to {self.metrics_filepath}: {e}"
            )
            raise


def main():
    parser = argparse.ArgumentParser(description="Run MOFA model")
    parser.add_argument("--config_path", type=str, default="/app/config_alldatasets.json", help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)
    os.makedirs(config["output_dir"], exist_ok=True)
    setup_logging(config["output_dir"])

    # Data information from config file
    datasets = load_datasets(args.config_path)
    data_concat = dataset_select(datasets_dict=datasets, data_type="mudata")

    try:
        for dataset_name, data_dict in data_concat.items():
            # Instantiate and run model
            model = MOFAModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            logger.info(f"Running MOFA model on dataset: {dataset_name}")
            # Run the model pipeline
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            logger.info(f"MOFA model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during MOFA model run: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
