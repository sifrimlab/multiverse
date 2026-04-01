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


def determine_valid_metrics(
    config: dict, dataset: ad.AnnData, requested_metrics: dict = None
):
    """Filters evaluation metrics based on dataset properties.

    Supervised metrics (e.g., ARI, NMI) are only valid if a cell type key is present.
    Batch correction metrics are only valid if multiple batches exist.

    Args:
        config (dict): The system configuration.
        dataset (ad.AnnData): The dataset to be evaluated.
        requested_metrics (dict, optional): A dictionary specifying the desired
            bio-conservation and batch-correction metrics.

    Returns:
        dict: A dictionary containing the lists of validated metrics.
    """
    if requested_metrics is None:
        requested_metrics = {
            "bio_conservation": ["isolated_labels", "nmi_ari_cluster_labels_leiden", "nmi_ari_cluster_labels_kmeans", "silhouette_label", "clisi_knn"],
            "batch_correction": ["bras", "ilisi_knn", "kbet_per_label", "graph_connectivity", "pcr_comparison"]
        }

    valid_metrics = {
        "bio_conservation": [],
        "batch_correction": []
    }

    batch_key = config.get("batch_key", "batch")
    label_key = config.get("cell_type_key")

    # Bio-conservation metrics generally require cell type labels.
    if label_key and label_key in dataset.obs.columns:
        valid_metrics["bio_conservation"] = requested_metrics["bio_conservation"]
    else:
        logger.warning(f"Label key {label_key} not found or not provided. Skipping supervised metrics.")
        # Remove metrics that strictly require labels.
        supervised_keywords = ["nmi", "ari", "isolated_labels", "clisi", "label"]
        valid_metrics["bio_conservation"] = [
            m for m in requested_metrics["bio_conservation"]
            if not any(kw in m.lower() for kw in supervised_keywords)
        ]

    # Batch correction metrics require at least two distinct batches to be meaningful.
    if batch_key in dataset.obs.columns:
        num_batches = dataset.obs[batch_key].nunique()
        if num_batches > 1:
            valid_metrics["batch_correction"] = requested_metrics["batch_correction"]
        else:
            logger.warning(f"Only one batch found ({num_batches}). Skipping batch-correction metrics.")
            valid_metrics["batch_correction"] = []
    else:
        logger.warning(f"Batch key {batch_key} not found. Skipping batch-correction metrics.")
        valid_metrics["batch_correction"] = []

    return valid_metrics


def aggregate_results(model_status: dict, output_dir: str):
    """Aggregates metrics from successful model runs into a single JSON file.

    Args:
        model_status (dict): Mapping of model names to their execution status.
        output_dir (str): The directory containing model outputs.

    Returns:
        dict: The aggregated results dictionary.
    """
    final_results = {}

    for model_name, status in model_status.items():
        if status != "success":
            continue

        model_metrics_path = os.path.join(output_dir, model_name, "metrics.json")
        if os.path.exists(model_metrics_path):
            with open(model_metrics_path, "r") as f:
                final_results[model_name] = json.load(f)
        else:
            final_results[model_name] = {"status": "success", "info": "Metrics file not found, but model succeeded."}

    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=4)

    logger.info(f"Aggregated results saved to {results_file}")
    return final_results


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
        config_path: str,
        is_gridsearch: bool = False,
    ):
        """Initializes the Evaluator.

        Args:
            dataset (ad.AnnData): The dataset containing latent representations.
            dataset_name (str): Name of the dataset.
            config_path (str): Path to the configuration file.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
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
        """Loads latent embeddings for each model from the output directory.

        Iterates through the models defined in the configuration and attempts
        to load their saved embeddings from HDF5 files.
        """
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
