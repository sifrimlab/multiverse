import random
from typing import Union

import anndata as ad
import numpy as np
import scvi
import pandas as pd
from sklearn.metrics import silhouette_score
from ..data_utils import anndata_concatenate
from ..logging_utils import get_logger
from ..utils import get_device
from .base import ModelFactory
from .metrics_utils import scvi_history_to_dict
from .runtime_io import (
    build_model_config,
    load_input_mudata,
    load_job_spec,
    setup_container_logging,
)

logger = get_logger(__name__)

class MultiVIModel(ModelFactory):
    """MultiVI model wrapper from the `scvi-tools` library.

    Designed for joint analysis of single-cell RNA and ATAC sequencing data.

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
        """Initializes the MultiVIModel.

        Args:
            dataset (ad.AnnData): Concatenated RNA and ATAC AnnData object.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'multivi' configuration is missing or 'feature_types' are not defined.
        """
        logger.info("Initializing MultiVI Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="multivi", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        multivi_params= self.model_params.get(self.model_name)

        self.device = multivi_params.get("device")
        self.max_epochs = multivi_params.get("max_epochs")
        self.learning_rate = multivi_params.get("learning_rate")
        self.umap_color_type = multivi_params.get("umap_color_type")
        self.torch_device = "cpu"
        self.latent_dimensions = multivi_params.get("latent_dimensions")
        self.umap_random_state = multivi_params.get("umap_random_state")
        self.torch_device = get_device(self.device)

        if "feature_types" in self.dataset.var.keys():
            try:
                self.dataset = self.dataset[:, self.dataset.var["feature_types"].argsort()].copy()
                if "Protein Expression" in self.dataset.var["feature_types"].unique():
                    protein_indices = self.dataset.var["feature_types"] == "protein expression"
                    protein_expression = self.dataset.X[:, protein_indices]
                    protein_names = self.dataset.var_names[protein_indices]
                    protein_expression_df = pd.DataFrame(protein_expression,
                                     index=self.dataset.obs_names,
                                     columns=protein_names)
                    self.dataset.obsm["protein_expression"] = protein_expression_df
                    scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key="protein_expression")
                else:
                    scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key=None)

                self.model = scvi.model.MULTIVI(self.dataset,
                                                n_genes=(self.dataset.var["feature_types"] == "Gene Expression").sum(),
                                                n_regions=(self.dataset.var["feature_types"] == "Peaks").sum(),
                                                )
            except Exception as e:
                logger.error(f"Something is wrong in MultiVI initialization: {e}")
                raise
        else:
            raise ValueError("MultiVI initialization needs 'feature_types' in variable keys to setup genes (RNA-seq) and genomic regions (ATAC-seq)!")

    def train(self):
        """Trains the MultiVI model using stochastic variational inference."""
        logger.info("Training MultiVI Model")
        try:
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            logger.info("Multivi training completed.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        """Evaluates the MultiVI model using the silhouette score.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        history = scvi_history_to_dict(
            self.model.history if hasattr(self.model, "history") else None
        )
        if self.latent_key in self.dataset.obsm:
            latent = self.dataset.obsm[self.latent_key]
            if (requested is None or "silhouette_score" in requested):
                if self.umap_color_type and self.umap_color_type in self.dataset.obs:
                    labels = self.dataset.obs[self.umap_color_type]
                    silhouette = silhouette_score(latent, labels)
                    logger.info(f"Silhouette Score (MultiVI): {silhouette}")
                    metrics["silhouette_score"] = float(silhouette)
                else:
                    logger.warning("Labels not found for clustering evaluation.")
        else:
            logger.warning("Latent representation (X_multivi) not found.")

        filtered_history = {}
        for key, series in history.items():
            if requested is None or key in requested:
                filtered_history[key] = series
        self.write_metrics(metrics, history=filtered_history or None)

def main():
    setup_container_logging()
    job_spec = load_job_spec()
    config = build_model_config(model_name="multivi", job_spec=job_spec)
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
        model = MultiVIModel(
            dataset=data_concat,
            dataset_name=dataset_name,
            config_path=config,
        )
        logger.info(f"Running MultiVI model on dataset: {dataset_name}")
        model.train()
        model.save_latent()
        model.umap()
        model.evaluate_model()
        logger.info(f"MultiVI model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during MultiVI model run: {e}")
        raise

if __name__ == "__main__":
    main()
