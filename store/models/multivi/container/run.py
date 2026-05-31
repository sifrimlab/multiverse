"""MultiVI container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import json
import os
import random
from typing import Union

import numpy as np
import pandas as pd
import scvi
import anndata as ad
from sklearn.metrics import silhouette_score

from mvr_worker import (
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    replay_history,
    setup_container_logging,
    ModelFactory,
    get_device,
    scvi_history_to_dict,
    preprocess_mudata,
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

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="multivi",
            is_gridsearch=is_gridsearch,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        multivi_params = self.model_params.get(self.model_name)

        self.device = multivi_params.get("device")
        self.max_epochs = multivi_params.get("max_epochs")
        self.learning_rate = multivi_params.get("learning_rate")
        self.umap_color_type = multivi_params.get("umap_color_type")
        self.latent_dimensions = multivi_params.get("latent_dimensions")
        self.umap_random_state = multivi_params.get("umap_random_state")
        self.torch_device = get_device(self.device)
        self.dataset_name = dataset_name

        if "feature_types" not in self.dataset.var.keys():
            raise ValueError(
                "MultiVI initialization needs 'feature_types' in variable keys to setup genes (RNA-seq) and genomic regions (ATAC-seq)!"
            )

        try:
            self.dataset = self.dataset[
                :, self.dataset.var["feature_types"].argsort()
            ].copy()
            if "Protein Expression" in self.dataset.var["feature_types"].unique():
                protein_indices = (self.dataset.var["feature_types"] == "Protein Expression").values
                protein_col_idx = np.where(protein_indices)[0]
                protein_names = self.dataset.var_names[protein_indices]
                raw_protein = self.dataset.X[:, protein_col_idx]
                if hasattr(raw_protein, "toarray"):
                    raw_protein = raw_protein.toarray()
                protein_expression_df = pd.DataFrame(
                    raw_protein,
                    index=self.dataset.obs_names,
                    columns=protein_names,
                )
                self.dataset.obsm["protein_expression"] = protein_expression_df
                scvi.model.MULTIVI.setup_anndata(
                    self.dataset, protein_expression_obsm_key="protein_expression", batch_key="modality"
                )
            else:
                scvi.model.MULTIVI.setup_anndata(
                    self.dataset, protein_expression_obsm_key=None, batch_key="modality"
                )

            self.model = scvi.model.MULTIVI(
                self.dataset,
                n_genes=(self.dataset.var["feature_types"] == "Gene Expression").sum(),
                n_regions=(self.dataset.var["feature_types"] == "Peaks").sum(),
                n_latent=self.latent_dimensions,
            )
            self.model.to_device(self.torch_device)
        except Exception as e:
            logger.error(f"Something is wrong in MultiVI initialization: {e}")
            raise

    def train(self):
        """Trains the MultiVI model using stochastic variational inference."""
        logger.info("Training MultiVI Model")
        try:
            self.model.train(
                max_epochs=self.max_epochs,
                lr=self.learning_rate,
               
            )
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            logger.info("MultiVI training completed.")
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
            if requested is None or "silhouette_score" in requested:
                if self.umap_color_type and self.umap_color_type in self.dataset.obs:
                    labels = self.dataset.obs[self.umap_color_type]
                    if np.unique(labels).shape[0] < 2:
                        logger.warning("Silhouette score cannot be computed with less than 2 clusters.")
                        metrics["silhouette_score"] = 0
                    else:
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
        filtered_history = replay_history(
            filtered_history,
            output_dir=OUTPUT_DIR,
            run_name=f"{self.dataset_name}-multivi-{os.path.basename(OUTPUT_DIR)}",
        )
        self.write_metrics(metrics, history=filtered_history or None)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("multivi", job_spec, OUTPUT_DIR)
    
    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed

    mudata_obj = load_input_mudata()
    modalities = mudata_obj.mod.keys()
    # TODO: make preprocessing not hardcoded but GUI based
    # TODO: make cell type and batch key not hardcoded
    config["preprocess_params"] = {
        "n_top_genes": 1000,
        "scale": {modality: False for modality in modalities},
        "normalization_target_sum": None,
        "log_normalization": False,
    }
    mudata_obj = preprocess_mudata(
        mudata_obj,
        config["preprocess_params"],
        cell_type_key="cell_type",
        batch_key="batch",
    )
    dataset_name = job_spec.get("dataset_slug", "dataset")
    data_concat = anndata_concatenate(
        mdata=mudata_obj,
        selected_modalities= ["rna","atac","adt"],
        cell_type_key="cell_type",
        batch_key="batch",
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
