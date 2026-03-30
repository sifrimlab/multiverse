import argparse
import os
import json
import h5py
import scanpy as sc
import anndata as ad
import muon as mu
import matplotlib.pyplot as plt
import numpy as np
from ..config import load_config
from ..logging_utils import get_logger, setup_logging
from ..data_utils import load_datasets, dataset_select

from .base import ModelFactory

logger = get_logger(__name__)

class MOFAModel(ModelFactory):
    """MOFA Model implementation."""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False):
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
        """
        Train the MOFA model.
        """
        logger.info("Training MOFA+ Model")
        try:
            mu.tl.mofa(
                data=self.dataset, n_factors=self.n_factors, gpu_mode=self.gpu_mode
            )
            self.dataset.obsm[self.latent_key] = self.dataset.obsm["X_mofa"]
            logger.info("MOFA training completed.")

            # Debugging output
            # logger.debug(f"Keys in dataset.uns['mofa']: {self.dataset.uns.get('mofa', {}).keys()}")

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
        """
        Compute explained variance for MOFA factors.
        """
        try:
            factors = self.dataset.obsm[self.latent_key]  # Extract latent factors
            # logger.debug(f"Latent factors (X_mofa) shape: {factors.shape}")

            # Compute total variance from raw data across modalities
            total_variance = 0
            for modality in self.dataset.mod.values():
                if hasattr(modality.X, "toarray"):
                    modality_data = (
                        modality.X.toarray()
                    )  # Convert sparse to dense if needed
                else:
                    modality_data = modality.X
                total_variance += np.var(modality_data, axis=0).sum()

            # logger.debug(f"Total variance from all modalities: {total_variance}")

            # Variance explained by factors
            factor_variances = np.var(factors, axis=0)
            # logger.debug(f"Factor variances: {factor_variances}")

            explained_variance_ratio = factor_variances / total_variance
            # logger.debug(f"Explained variance ratio per factor: {explained_variance_ratio}")
            return explained_variance_ratio

        except Exception as e:
            logger.error(f"Error computing explained variance: {e}")
            return []
    def _compute_explained_variance(self):
        """
        Compute explained variance for MOFA factors.
        """
        try:
            factors = self.dataset.obsm[self.latent_key]  # Extract latent factors
            # logger.debug(f"Latent factors (X_mofa) shape: {factors.shape}")

            # Compute total variance from raw data across modalities
            total_variance = 0
            for modality in self.dataset.mod.values():
                if hasattr(modality.X, "toarray"):
                    modality_data = (
                        modality.X.toarray()
                    )  # Convert sparse to dense if needed
                else:
                    modality_data = modality.X
                total_variance += np.var(modality_data, axis=0).sum()

            # logger.debug(f"Total variance from all modalities: {total_variance}")

            # Variance explained by factors
            factor_variances = np.var(factors, axis=0)
            # logger.debug(f"Factor variances: {factor_variances}")

            explained_variance_ratio = factor_variances / total_variance
            # logger.debug(f"Explained variance ratio per factor: {explained_variance_ratio}")
            return explained_variance_ratio

        except Exception as e:
            logger.error(f"Error computing explained variance: {e}")
            return []
    def evaluate_model(self):
        """
        Evaluate the trained MOFA+ model based on explained variance.
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
