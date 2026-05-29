"""PCA container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import random
from typing import Union

import numpy as np
import scanpy as sc
import anndata as ad


from mvr_worker import (
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    save_embeddings,
    save_umap,
    setup_container_logging,
    ModelFactory,
)

logger = get_logger(__name__)

class PCAModel(ModelFactory):
    """Principal Component Analysis (PCA) model wrapper.

    Performs PCA on concatenated multimodal data to generate low-dimensional
    latent embeddings.

    Attributes:
        n_components (int): The number of principal components to calculate.
        device (str): Computation device (e.g., "cpu").
        gpu_mode (bool): Flag indicating if GPU acceleration is used.
    """

    def __init__(
        self,
        dataset: ad.AnnData,
        dataset_name: str,
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
    ):
        """Initializes the PCAModel.

        Args:
            dataset (ad.AnnData): Concatenated multimodal AnnData object.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'pca' configuration is not found in the model parameters.
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
        """Calculates principal components on the dataset."""
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
        """Evaluates the PCA model by calculating the total explained variance.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        if hasattr(self, "variance_ratio"):
            if requested is None or "total_variance" in requested:
                total_variance = sum(self.variance_ratio)
                logger.info(f"Total Variance Explained: {total_variance}")
                metrics["total_variance"] = total_variance
        else:
            logger.warning("PCA variance ratio not available in the model.")

        logger.info(f"Evaluation metrics: {metrics}")
        self.write_metrics(metrics)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    logger.info("PCA container run script started.")
    job_spec = load_job_spec()
    config = build_model_config("pca", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)

    params = config["model"].get("pca", {})
    n_components = params.get("n_components", 50)
    try:

        mudata_obj = load_input_mudata()
        dataset_name = job_spec.get("dataset_name", "dataset")
        data_concat = anndata_concatenate(
        list_anndata=[mudata_obj[modality] for modality in mudata_obj.mod.keys()],
        list_modality=list(mudata_obj.mod.keys()),
        )
    except Exception as e:
        logger.error(f"Failed to load and concatenate input data: {e}")
        raise

    logger.info(f"Running PCA with n_components={n_components}")
    try:
        pca_model = PCAModel(
            dataset=data_concat,
            dataset_name=dataset_name,
            config_path=config,
        )
        pca_model.train()
        pca_model.save_latent()
        pca_model.umap()
        pca_model.evaluate_model()
        logger.info(f"PCA model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during PCA model run: {e}")
        raise

if __name__ == "__main__":
    main()
