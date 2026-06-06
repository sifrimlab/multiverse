"""WNN container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""

import random
from typing import Union

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import scanpy as sc
from mvr_worker import (OUTPUT_DIR, ModelFactory, build_model_config,
                        get_logger, load_input_mudata, load_job_spec,
                        preprocess_mudata, resolve_labels_key_params,
                        resolve_preprocess_params, setup_container_logging)
from pyWNN.pyWNN import pyWNN

logger = get_logger(__name__)

class WNNModel(ModelFactory):
    """Weighted Nearest Neighbor (WNN) multi-modal integration model.

    Computes per-modality PCA, builds a WNN graph weighting each modality by
    its local predictive power, then derives a WNN-UMAP embedding.

    Attributes:
        n_neighbors (int): Nearest neighbors used for each modality's KNN graph.
        n_pcs_per_modality (int): PCA components computed per modality.
        umap_random_state (int): Random state for UMAP reproducibility.
    """

    def __init__(
        self,
        dataset: md.MuData,
        dataset_name: str,
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
        cell_type_key: str = "cell_type",
        batch_key: str = "batch",
    ):
        """Initializes the WNNModel.

        Args:
            dataset (md.MuData): Multi-modal MuData object.
            dataset_name (str): Name of the dataset.
            config_path: Path to the JSON configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
            cell_type_key (str): Key in .obs for cell type annotations.
            batch_key (str): Key in .obs for batch annotations.

        Raises:
            ValueError: If 'wnn' configuration is not found in the model parameters.
        """
        logger.info("Initializing WNN Model")

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="wnn",
            is_gridsearch=is_gridsearch,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        wnn_params = self.model_params.get(self.model_name)
        self.modality_1 = wnn_params.get("modality_1")
        self.modality_2 = wnn_params.get("modality_2")
        self.n_neighbors = wnn_params.get("n_neighbors")
        self.n_pcs_per_modality = wnn_params.get("n_pcs_per_modality")
        self.umap_random_state = wnn_params.get("umap_random_state")
        self.umap_color_type = wnn_params.get("umap_color_type")

        available = list(dataset.mod.keys()) if hasattr(dataset, "mod") else []
        for mod in (self.modality_1, self.modality_2):
            if mod not in available:
                raise ValueError(
                    f"Modality '{mod}' not found in dataset. Available: {available}"
                )

        logger.info(
            f"WNN initialized with {self.dataset_name}, "
            f"modalities=[{self.modality_1}, {self.modality_2}], "
            f"{self.n_neighbors} neighbors, {self.n_pcs_per_modality} PCs/modality."
        )

    def train(self):
        """Computes per-modality PCA, WNN graph, and WNN-UMAP.

        Stores the 2-D WNN-UMAP coordinates in ``self.dataset.obsm[self.latent_key]``
        and per-modality mean weights in ``self._modality_weights``.
        """


        logger.info("Training WNN Model")
        mdata = self.dataset
        modalities = [self.modality_1, self.modality_2]

        obsm_keys = []
        n_pcs_list = []
        for mod_name in modalities:
            mod = mdata[mod_name]
            n_pcs = min(self.n_pcs_per_modality, mod.n_vars - 1)
            use_hvg = "highly_variable" in mod.var.columns
            sc.tl.pca(mod, n_comps=n_pcs, use_highly_variable=use_hvg, svd_solver="arpack")
            obsm_key = f"{mod_name.upper()}_PCA"
            obsm_keys.append(obsm_key)
            n_pcs_list.append(n_pcs)
            logger.info(f"PCA: modality='{mod_name}', n_pcs={n_pcs}, hvg={use_hvg}")

        # pyWNN requires a joint AnnData with per-modality embeddings in .obsm
        joint = ad.AnnData(obs=mdata.obs.copy())
        for mod_name, obsm_key, n_pcs in zip(modalities, obsm_keys, n_pcs_list):
            joint.obsm[obsm_key] = mdata[mod_name].obsm["X_pca"][:, :n_pcs]

        logger.info(
            f"Running pyWNN: reps={obsm_keys}, npcs={n_pcs_list}, "
            f"n_neighbors={self.n_neighbors}"
        )
        wnn_obj = pyWNN(
            joint,
            reps=obsm_keys,
            npcs=n_pcs_list,
            n_neighbors=self.n_neighbors,
            seed=self.umap_random_state,
        )
        joint = wnn_obj.compute_wnn(joint)

        sc.tl.umap(joint, neighbors_key="WNN", random_state=self.umap_random_state)

        # WNN-UMAP becomes the canonical latent embedding
        mdata.obsm[self.latent_key] = joint.obsm["X_umap"]

        # Store per-modality weights in .obs for downstream inspection
        for i, mod_name in enumerate(modalities):
            mdata.obs[f"WNN_weight_{mod_name}"] = wnn_obj.weights[i]

        self._modality_weights = {
            mod_name: float(np.mean(wnn_obj.weights[i]))
            for i, mod_name in enumerate(modalities)
        }

        logger.info("WNN training completed.")
        logger.info(f"Mean modality weights: {self._modality_weights}")

    def umap(self):
        """Saves a UMAP plot using the WNN-UMAP already computed in train().

        Overrides the base implementation, which would incorrectly attempt to
        re-run neighbor search on the 2-D latent coordinates.
        """
        logger.info("Saving WNN UMAP visualization")
        try:
            # Copy to X_umap so sc.pl.umap can find it
            self.dataset.obsm["X_umap"] = self.dataset.obsm[self.latent_key]

            if self.umap_color_type in self.dataset.obs:
                sc.pl.umap(self.dataset, color=self.umap_color_type, show=False)
            else:
                logger.warning(
                    f"UMAP color key '{self.umap_color_type}' not found in .obs. "
                    "Plotting without color."
                )
                sc.pl.umap(self.dataset, show=False)

            plt.savefig(self.umap_filename, bbox_inches="tight")
            plt.close()
            logger.info(f"WNN UMAP plot saved to {self.umap_filename}")
        except Exception as e:
            logger.error(f"Error during WNN UMAP visualization: {e}")
            raise

    def evaluate_model(self):
        """Reports mean WNN weight per modality as model metrics.

        A weight close to 1/N (uniform) means all modalities contribute equally;
        a higher weight signals that modality dominates the local neighborhood.
        """
        requested = self.config_dict.get("metrics", {}).get("model_metrics")
        metrics = {}
        if hasattr(self, "_modality_weights"):
            for mod_name, mean_weight in self._modality_weights.items():
                key = f"mean_weight_{mod_name}"
                if requested is None or key in requested:
                    metrics[key] = mean_weight
                    logger.info(f"{key}: {mean_weight:.4f}")
        else:
            logger.warning("WNN modality weights not available.")

        self.write_metrics(metrics)


def main() -> None:
    """Container entry: load job spec and data, preprocess, train, write outputs."""
    setup_container_logging(OUTPUT_DIR)
    logger.info("WNN container run script started.")
    job_spec = load_job_spec()
    config = build_model_config("wnn", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)

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
                "scale": {mod: False for mod in modalities},
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
        logger.error(f"Failed to load and preprocess input data: {e}")
        raise

    dataset_name = job_spec.get("dataset_slug", "dataset")

    try:
        model = WNNModel(
            dataset=mdata,
            dataset_name=dataset_name,
            config_path=config,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
        )
        logger.info(f"Running WNN model on dataset: {dataset_name}")
        model.train()
        model.save_latent()
        model.umap()
        model.evaluate_model()
        logger.info(f"WNN model run for {dataset_name} completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during WNN model run: {e}")
        raise


if __name__ == "__main__":
    main()
