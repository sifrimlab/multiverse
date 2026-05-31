"""TotalVI container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import random
from typing import Union

import numpy as np
import scvi
import anndata as ad
import pandas as pd
from mvr_worker import (
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    setup_container_logging,
    ModelFactory,
    get_device,
    scvi_history_to_dict,
    preprocess_mudata,
)

logger = get_logger(__name__)

class TotalVIModel(ModelFactory):
    """TotalVI model wrapper from the `scvi-tools` library.

    Designed for joint analysis of single-cell RNA and protein data.

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
        """Initializes the TotalVIModel.

        Args:
            dataset (ad.AnnData): Concatenated RNA and Protein AnnData object.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
                Defaults to False.

        Raises:
            ValueError: If 'totalvi' configuration is missing.
        """
        logger.info("Initializing TotalVI Model")

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="totalvi",
            is_gridsearch=is_gridsearch,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        totalvi_params = self.model_params.get(self.model_name)
        self.device = totalvi_params.get("device")
        self.max_epochs = totalvi_params.get("max_epochs")
        self.learning_rate = totalvi_params.get("learning_rate")
        self.umap_color_type = totalvi_params.get("umap_color_type")
        self.torch_device = "cpu"
        self.latent_dimensions = totalvi_params.get("latent_dimensions")
        self.umap_random_state = totalvi_params.get("umap_random_state")
        self.torch_device = get_device(self.device)

        if "feature_types" not in self.dataset.var.keys():
            raise ValueError(
                "TotalVI initialization needs 'feature_types' in variable keys to setup genes (RNA-seq) and genomic regions (ATAC-seq)!"
            )
        if "Protein Expression" not in self.dataset.var["feature_types"].unique():
            raise ValueError(
                "No protein expression data found in the dataset. TotalVI requires protein expression data for training."
            )

        try:
            self.dataset = self.dataset[
                :, self.dataset.var["feature_types"].argsort()
            ].copy()
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
            scvi.model.TOTALVI.setup_anndata(
                self.dataset,
                protein_expression_obsm_key="protein_expression",
                batch_key="batch",
            )
        except Exception as e:
            logger.error(f"Something is wrong in TotalVI initialization: {e}")
            raise

        self.model = scvi.model.TOTALVI(self.dataset)
        self.model.to_device(self.torch_device)

    def train(self):
        """Trains the TotalVI model using variational inference."""
        logger.info("Training TotalVI Model")
        try:
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            logger.info("TotalVI training completed.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        """Evaluates the TotalVI model.

        Writes the resulting metrics to a JSON file.

        Raises:
            IOError: If the metrics file cannot be written.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        history = scvi_history_to_dict(
            self.model.history if hasattr(self.model, "history") else None
        )
        if hasattr(self.model, "history") and self.model.history:
            if requested is None or "elbo_train" in requested:
                if "elbo_train" in history:
                    metrics["elbo_train"] = history["elbo_train"][-1]
            if requested is None or "reconstruction_loss_train" in requested:
                if "reconstruction_loss_train" in history:
                    metrics["reconstruction_loss_train"] = history[
                        "reconstruction_loss_train"
                    ][-1]
        filtered_history = {}
        for key, series in history.items():
            if requested is None or key in requested:
                filtered_history[key] = series
        self.write_metrics(metrics, history=filtered_history or None)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("totalvi", job_spec, OUTPUT_DIR)
    
    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed
    mudata_obj = load_input_mudata()
    # TODO: make preprocessing not hardcoded but GUI based
    # TODO: make cell type and batch key not hardcoded
    modalities = list(mudata_obj.mod.keys())
    config["preprocess_params"] = {
        "n_top_genes": 1000,
        "scale": {mod: False for mod in modalities},  # Skip scaling to preserve count-based nature of the data for TotalVI
        "normalization_target_sum": None,  # No normalization to preserve count-based nature of the data for TotalVI
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
        selected_modalities=["rna", "adt"], # only concatenate RNA and Protein modalities for TotalVI
        cell_type_key="cell_type",
        batch_key="batch",
    )

    try:
        model = TotalVIModel(
            dataset=data_concat,
            dataset_name=dataset_name,
            config_path=config,
        )
        logger.info(f"Running TotalVI model on dataset: {dataset_name}")
        model.train()
        model.save_latent()
        model.umap()
        model.evaluate_model()
        logger.info(f"TotalVI model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during TotalVI model run: {e}")
        raise

if __name__ == "__main__":
    main()
