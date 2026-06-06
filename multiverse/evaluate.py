"""Post-training evaluation metrics for integration model outputs."""

import argparse
import json
import os
from typing import Union

import muon as mu
import anndata as ad
import h5py
import numpy as np
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

from mvr_worker import (
    anndata_concatenate,
    get_logger,
    load_config,
    preprocess_mudata,
    sanitize_nan_inf,
)

logger = get_logger(__name__)


class Evaluator:
    """A class for evaluating data integration models using scIB metrics.

    Attributes:
        dataset (ad.AnnData): The dataset containing model embeddings.
        dataset_name (str): The name of the dataset.
        output_dir (str): Directory for saving evaluation results.
        metrics_filepath (str): Path to the final metrics JSON file.
    """

    def __init__(
        self,
        dataset: ad.AnnData,
        dataset_name: str,
        model_configs: dict,
        output_dir: str,
    ):
        """Initializes the Evaluator.

        Args:
            dataset (ad.AnnData): The dataset containing latent representations.
            dataset_name (str): Name of the dataset.
            model_configs (dict): A dictionary containing model configurations.
            output_dir (str): Directory for saving evaluation results.
        """
        logger.info("Initializing Evaluator")
        self.model_configs = model_configs
        self.model_params = model_configs
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.metrics_filepath = os.path.join(
            self.output_dir,
            "evaluation_metrics.json",
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.latent_keys = []

        logger.info(f"Evaluator initialized for {self.dataset_name}")

    def load_embeddings(self):
        """Loads latent embeddings for each model from the output directory.

        Iterates through the models defined in the configuration and attempts
        to load their saved embeddings from HDF5 files.
        """
        for model_config in self.model_configs:
            logger.info(f"Loading embeddings for model: {model_config['model_name']}")
            if os.path.exists(model_config["output_embedding_path"]):
                latent_key = f"X_{model_config['model_name']}"
                self.latent_keys.append(latent_key)
                self.dataset.obsm[latent_key] = h5py.File(
                    model_config["output_embedding_path"], "r"
                )["latent"][:]
            else:
                logger.warning(f"Embeddings file not found: {model_config['output_embedding_path']}")
                logger.warning(
                    f"Embeddings file for model {model_config['model_name']} not found. Skipping."
                )
        logger.info(f"Loaded latent embeddings for: {self.latent_keys}")

    def evaluate_models(self, batch_key: str = "batch", label_key: str = "cell_type"):
        """Runs the scIB-metrics benchmark suite on all loaded model embeddings.

        Calculates bio-conservation and batch-correction metrics and saves
        the results to a JSON file and a summary table plot.

        Args:
            batch_key (str): The observation key for batch labels. Defaults to "batch".
            label_key (str): The observation key for cell type labels. Defaults to "cell_type".

        Returns:
            dict: The calculated metrics.
        """
        logger.info("Evaluating model with scib-metrics.")

        batch_correction_metrics_requested = True
        bio_conservation_metrics_requested = True
        if (
            batch_key not in self.dataset.obs.columns
            or self.dataset.obs[batch_key].nunique() < 2
        ):
            logger.warning(
                f"Batch key '{batch_key}' not found in .obs, assigning dummy batch labels for batch correction metrics "
            )
            rng = np.random.default_rng()
            self.dataset.obs[batch_key] = rng.choice(
                [f"batch_{i}" for i in range(10)], size=self.dataset.n_obs
            )

            batch_correction_metrics_requested = True

        if label_key not in self.dataset.obs.columns or self.dataset.obs[label_key].nunique() < 2:
            logger.warning(
                f"Label key '{label_key}' not found in .obs, assigning dummy cell type labels for bio-conservation metrics"
            )
            bio_conservation_metrics_requested = True
            rng = np.random.default_rng()
            self.dataset.obs[label_key] = rng.choice(
                [f"cell_type_{i}" for i in range(10)], size=self.dataset.n_obs
            )

        bm = Benchmarker(
            self.dataset,
            progress_bar=False,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=self.latent_keys,
            bio_conservation_metrics=BioConservation(
                isolated_labels=bio_conservation_metrics_requested,
                nmi_ari_cluster_labels_leiden=bio_conservation_metrics_requested,
                nmi_ari_cluster_labels_kmeans=bio_conservation_metrics_requested,
                silhouette_label=bio_conservation_metrics_requested,
                clisi_knn=bio_conservation_metrics_requested,
            ),
            batch_correction_metrics=BatchCorrection(
                bras=batch_correction_metrics_requested,
                ilisi_knn=batch_correction_metrics_requested,
                kbet_per_label=batch_correction_metrics_requested,
                graph_connectivity=batch_correction_metrics_requested,
                pcr_comparison=batch_correction_metrics_requested,
            ),
        )

        bm.benchmark()
        results_df = bm.get_results(min_max_scale=False)
        bm.plot_results_table(min_max_scale=False, show=False, save_dir=self.output_dir)
        if results_df.empty:
            logger.warning(f"No results found for {self.dataset_name}.")
            return

        metrics = sanitize_nan_inf(results_df.to_dict("dict"))
        logger.info(f"Evaluation metrics: {metrics}")
        try:
            with open(self.metrics_filepath, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_filepath}")
        except IOError as e:
            logger.error(
                f"Could not write metrics file to {self.metrics_filepath}: {e}"
            )
            raise
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Evaluator")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/app/config_alldatasets.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    model_configs_per_dataset: dict = {}
    dataset_configs: dict = {}

    for config_members in config["members"]:
        dataset_name = config_members["dataset_slug"]
        #TODO: use resolve_labels_key_params to get batch_key and cell_type_key, with fallbacks to "batch" and "cell_type" respectively (issue #22)
        batch_key = config_members.get("batch_key", "batch")
        label_key = config_members.get("cell_type_key", "cell_type")

        dataset_config = {
            "dataset_name": dataset_name,
            "data_path": config_members["dataset_path_resolved"],
            "modalities": config_members["job"]["omics_available"],
            "batch_key": batch_key,
            "label_key": label_key,
        }
        model_config = {
            "model_name": config_members["model_slug"],
            "output_embedding_path": os.path.join(
                config_members["artifact_dir"],
                "embeddings.h5",
            ),
        }

        if dataset_name in model_configs_per_dataset:
            model_configs_per_dataset[dataset_name].append(model_config)
        else:
            model_configs_per_dataset[dataset_name] = [model_config]

        if dataset_name in dataset_configs:
            logger.warning(
                f"Duplicate dataset name '{dataset_name}' found in config. Overwriting previous entry."
            )
        dataset_configs[dataset_name] = dataset_config

    for dataset_name, model_configs in model_configs_per_dataset.items():
        ds_cfg = dataset_configs[dataset_name]
        batch_key = ds_cfg["batch_key"]
        label_key = ds_cfg["label_key"]

        mudata_obj = mu.read_h5mu(ds_cfg["data_path"])
        modalities = list(mudata_obj.mod.keys())

        #TODO: use resolve_preprocess_params to get preprocessing parameters from the job spec, with fallbacks to these built-in defaults when unspecified (issue #22)
        mudata_obj = preprocess_mudata(
            mudata_obj,
            {
                "n_top_genes": 2000,
                "scale": {modality: False for modality in modalities},
                "normalization_target_sum": None,
                "log_normalization": False,
            },
            cell_type_key=label_key,
            batch_key=batch_key,
        )
        data_concat = anndata_concatenate(
            mdata=mudata_obj,
            selected_modalities=modalities,
            cell_type_key=label_key,
            batch_key=batch_key,
        )

        output_dir = os.path.join(
            config["output_dir"],
            "evaluation",
            config["launch_id"],
            f"dataset_{dataset_name}",
        )

        evaluator = Evaluator(
            dataset=data_concat,
            dataset_name=dataset_name,
            model_configs=model_configs,
            output_dir=output_dir,
        )
        evaluator.load_embeddings()
        evaluator.evaluate_models(batch_key=batch_key, label_key=label_key)


if __name__ == "__main__":
    main()
