"""MOFA container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""

import random
from typing import Union

import anndata as ad
import muon as mu
import numpy as np
from multiverse.worker import (OUTPUT_DIR, ModelFactory, build_model_config,
                        get_logger, load_input_mudata, load_job_spec,
                        preprocess_mudata, resolve_labels_key_params,
                        resolve_preprocess_params, setup_container_logging)

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
        cell_type_key: str = "cell_type",
        batch_key: str = "batch",  
    ):
        """Initializes the MOFAModel.

        Args:
            dataset (ad.AnnData): The input dataset (MuData-derived AnnData).
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.
            cell_type_key (str): Key in .obs for cell type annotations. Defaults to "cell_type".
            batch_key (str): Key in .obs for batch annotations. Defaults to "batch".

        Raises:
            ValueError: If 'mofa' configuration is not found in the model parameters.
        """
        logger.info("Initializing MOFA Model")

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="mofa",
            is_gridsearch=is_gridsearch,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        mofa_params = self.model_params.get(self.model_name)

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


def main() -> None:
    """Container entry: load job spec and data, preprocess, train, write outputs.

    ``cell_type_key`` and ``batch_key`` are fixed to ``cell_type`` / ``batch`` until
    job-spec overrides exist.
    """
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("mofa", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    dataset_name = job_spec.get("dataset_slug", "dataset")
    label_keys = resolve_labels_key_params(job_spec)
    cell_type_key = label_keys["cell_type_key"]
    batch_key = label_keys["batch_key"]
    try:
        mdata = load_input_mudata()
        modalities = list(mdata.mod.keys())
        config["preprocess_params"] = resolve_preprocess_params(
            job_spec,
            modalities,
            {
                "n_top_genes": 1000,
                "scale": {modality: True for modality in modalities},
                "normalization_target_sum": 1e4,
                "log_normalization": True,
            },
        )
        mdata = preprocess_mudata(
            mdata,
            config["preprocess_params"],
            cell_type_key=cell_type_key,
            batch_key=batch_key,
        )

    except Exception as e:
        logger.error(f"Failed to load and concatenate input data: {e}")
        raise

    try:
        model = MOFAModel(
            dataset=mdata,
            dataset_name=dataset_name,
            config_path=config,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
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
