import argparse
import os
import json
import scanpy as sc
import anndata as ad
from .base import ModelFactory
from ..config import load_config
from ..data_utils import load_datasets, dataset_select
from ..logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

class PCAModel(ModelFactory):
    """PCA implementation"""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False):
        """
        Initialize the PCA model with the specified parameters.
        Input data is AnnData object that was concatenated of multiple modality
        """
        logger.info("Initializing PCA Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="pca", is_gridsearch=is_gridsearch)

        # Check if model-specific params are present
        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        pca_params = self.model_params.get(self.model_name)

        # PCA parameters from config file
        self.n_components = pca_params.get("n_components")
        self.device = pca_params.get("device")
        self.gpu_mode = False # Cpu default mode
        self.umap_random_state = pca_params.get("umap_random_state")
        self.umap_color_type = pca_params.get("umap_color_type")

        logger.info(
            f"PCA initialized with {self.dataset_name}, {self.n_components} components."
        )

    def train(self):
        """Perform PCA on all modalities concatenated."""
        logger.info("Training PCA Model")

        if "highly_variable" in self.dataset.var.keys():
            sc.pp.pca(self.dataset, n_comps=self.n_components, use_highly_variable=True)
        else:
            sc.pp.pca(
                self.dataset, n_comps=self.n_components, use_highly_variable=False
            )

        self.dataset.obsm[self.latent_key] = self.dataset.obsm["X_pca"]
        self.variance_ratio = self.dataset.uns["pca"]["variance_ratio"]

        logger.info(f"Training PCA completed with {self.n_components} components")
        logger.info(f"Total variance explained: {sum(self.variance_ratio)}")


    def evaluate_model(self):
        """
        Evaluate the trained PCA model based on variance explained.
        """
        metrics = {}
        if hasattr(self, "variance_ratio"):
            total_variance = sum(self.variance_ratio)
            logger.info(f"Total Variance Explained: {total_variance}")
            metrics["total_variance"] = total_variance
        else:
            logger.warning("PCA variance ratio not available in the model.")

        logger.info(f"Evaluation metrics: {metrics}")

        try:
            with open(self.metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_filepath}")
        except IOError as e:
            logger.error(f"Could not write metrics file to {self.metrics_filepath}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Run PCA model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/app/config_alldatasets.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)
    os.makedirs(config["output_dir"], exist_ok=True)
    setup_logging(config["output_dir"])

    # Data information from config file
    datasets = load_datasets(args.config_path)
    data_concat = dataset_select(datasets_dict=datasets, data_type="concatenate")

    try:
        for dataset_name, data_dict in data_concat.items():
            # Instantiate and run model
            pca_model = PCAModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            # Run the model pipeline
            pca_model.train()
            pca_model.save_latent()
            pca_model.umap()
            pca_model.evaluate_model()

            logger.info(f"PCA model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during PCA model run: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
