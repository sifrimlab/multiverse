import argparse
import os
import json
import h5py
import anndata as ad
import numpy as np
from scib_metrics.benchmark import BioConservation, BatchCorrection, Benchmarker
from .config import load_config
from .data_utils import load_datasets, dataset_select
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


class Evaluator:
    """Evaluator implementation"""

    def __init__(
        self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False
    ):
        """
        Initialize the Evaluator.
        Input data is AnnData object that was concatenated of multiple modality
        """
        logger.info("Initializing Evaluator")
        self.config_dict = load_config(config_path=config_path)
        self.model_params = self.config_dict.get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(
            self.config_dict["output_dir"],
            self.dataset_name,
        )
        self.metrics_filepath = os.path.join(
            self.output_dir,
            "evaluation_metrics.json",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.models_list = [
            model_name for model_name in self.config_dict.get("model", {}).keys()
        ]
        self.latent_keys = []

        logger.info(f"Evaluator initialized for {self.dataset_name}")

    def load_embeddings(self):
        for model_name in self.models_list:
            logger.info(f"Loading embeddings for model: {model_name}")
            if os.path.exists(
                os.path.join(self.output_dir, model_name, "embeddings.h5")
            ):
                latent_key = f"X_{model_name}"
                self.latent_keys.append(latent_key)
                self.dataset.obsm[latent_key] = h5py.File(
                    os.path.join(self.output_dir, model_name, "embeddings.h5"), "r"
                )["latent"][:]
            else:
                logger.warning(f"Embeddings file for model {model_name} not found.")
        logger.info(f"Loaded latent embeddings for: {self.latent_keys}")

    def evaluate_models(self, batch_key="batch", label_key="cell_type"):
        """
        Evaluate the model using scib-metrics.
        """
        logger.info("Evaluating model with scib-metrics.")

        if batch_key not in self.dataset.obs.columns:
            logger.warning(
                f"Batch key '{batch_key}' not found in .obs, assigning samples randomly to two batches."
            )
            rng = np.random.default_rng()
            self.dataset.obs[batch_key] = rng.choice(
                ["batch_1", "batch_2"], size=self.dataset.n_obs
            )

        if label_key not in self.dataset.obs.columns:
            logger.warning(
                f"Label key '{label_key}' not found in .obs, skipping metrics that require it."
            )
            label_key = None

        bm = Benchmarker(
            self.dataset,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=self.latent_keys,
            bio_conservation_metrics=BioConservation(
                isolated_labels=True,
                nmi_ari_cluster_labels_leiden=True,
                nmi_ari_cluster_labels_kmeans=True,
                silhouette_label=True,
                clisi_knn=True,
            ),
            batch_correction_metrics=BatchCorrection(
                bras=True,
                ilisi_knn=True,
                kbet_per_label=True,
                graph_connectivity=True,
                pcr_comparison=True,
            ),
        )

        bm.benchmark()
        results_df = bm.get_results(min_max_scale=False)
        bm.plot_results_table(min_max_scale=False, show=False, save_dir=self.output_dir)
        if results_df.empty:
            logger.warning(f"No results found for {self.dataset_name}.")
            return

        metrics = results_df.to_dict("dict")
        logger.info(f"Evaluation metrics: {metrics}")
        try:
            with open(self.metrics_filepath, "w") as f:
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

    config = load_config(config_path=args.config_path)
    os.makedirs(config["output_dir"], exist_ok=True)
    setup_logging(config["output_dir"])

    # Data information from config file
    datasets = load_datasets(args.config_path)
    data_concat = dataset_select(datasets_dict=datasets, data_type="concatenate")

    try:
        for dataset_name, data_dict in data_concat.items():
            # Instantiate and run model
            evaluator = Evaluator(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            # Run the model pipeline
            evaluator.load_embeddings()
            evaluator.evaluate_models()

            logger.info(f"Evaluation completed for {dataset_name}.")
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise


if __name__ == "__main__":
    main()
