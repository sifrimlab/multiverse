import argparse
import json
import os
from typing import Union

import anndata as ad
import h5py
import numpy as np
from scib_metrics.benchmark import (BatchCorrection, Benchmarker,
                                    BioConservation)

from .config import load_config
from .data_utils import dataset_select, load_datasets
from .logging_utils import get_logger, setup_logging
from .tracking import sanitize_nan_inf

logger = get_logger(__name__)


def _warn_unrequested_result_columns(
    results_df, requested_metrics: dict | None
) -> None:
    if not requested_metrics:
        return
    requested = set()
    for values in requested_metrics.values():
        if isinstance(values, list):
            requested.update(str(v) for v in values)
    if not requested:
        return
    returned = set(map(str, results_df.columns))
    extra = returned - requested
    missing = requested - returned
    if extra:
        logger.warning(
            "Benchmark returned unrequested metric columns: %s", sorted(extra)
        )
    if missing:
        logger.warning(
            "Benchmark did not return requested metric columns: %s", sorted(missing)
        )


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
            "bio_conservation": [
                "isolated_labels",
                "nmi_ari_cluster_labels_leiden",
                "nmi_ari_cluster_labels_kmeans",
                "silhouette_label",
                "clisi_knn",
            ],
            "batch_correction": [
                "bras",
                "ilisi_knn",
                "kbet_per_label",
                "graph_connectivity",
                "pcr_comparison",
            ],
        }

    valid_metrics = {"bio_conservation": [], "batch_correction": []}

    batch_key = config.get("batch_key", "batch")
    label_key = config.get("cell_type_key")

    # Bio-conservation metrics generally require cell type labels.
    if label_key and label_key in dataset.obs.columns:
        valid_metrics["bio_conservation"] = requested_metrics["bio_conservation"]
    else:
        logger.warning(
            f"Label key {label_key} not found or not provided. Skipping supervised metrics."
        )
        # Remove metrics that strictly require labels.
        supervised_keywords = ["nmi", "ari", "isolated_labels", "clisi", "label"]
        valid_metrics["bio_conservation"] = [
            m
            for m in requested_metrics["bio_conservation"]
            if not any(kw in m.lower() for kw in supervised_keywords)
        ]

    # Batch correction metrics require at least two distinct batches to be meaningful.
    if batch_key in dataset.obs.columns:
        num_batches = dataset.obs[batch_key].nunique()
        if num_batches > 1:
            valid_metrics["batch_correction"] = requested_metrics["batch_correction"]
        else:
            logger.warning(
                f"Only one batch found ({num_batches}). Skipping batch-correction metrics."
            )
            valid_metrics["batch_correction"] = []
    else:
        logger.warning(
            f"Batch key {batch_key} not found. Skipping batch-correction metrics."
        )
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
            try:
                with open(model_metrics_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded:
                    final_results[model_name] = sanitize_nan_inf(loaded)
                else:
                    logger.warning(
                        "Metrics for %s were empty or not a JSON object; omitting.",
                        model_name,
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed reading metrics for %s from %s: %s",
                    model_name,
                    model_metrics_path,
                    exc,
                )
        else:
            final_results[model_name] = {
                "status": "success",
                "info": "Metrics file not found, but model succeeded.",
            }

    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_nan_inf(final_results), f, indent=4)

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
        config_path: Union[str, dict],
        is_gridsearch: bool = False,
    ):
        """Initializes the Evaluator.

        Args:
            dataset (ad.AnnData): The dataset containing latent representations.
            dataset_name (str): Name of the dataset.
            config_path: Path to the configuration file or an in-memory config dict.
            is_gridsearch (bool): Flag indicating if this is a grid search run.
        """
        logger.info("Initializing Evaluator")
        self.config_dict = load_config(config_path)
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
        requested_metrics = self.config_dict.get("metrics")
        _warn_unrequested_result_columns(results_df, requested_metrics)
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


def _mudata_to_evaluation_anndata(
    mdata,
    *,
    batch_key: str | None,
    label_key: str | None,
) -> "ad.AnnData":
    """Build a minimal AnnData from a MuData suitable for scIB evaluation.

    Top-level mdata.obs is used as the base. Any declared batch/label keys that
    are absent from top-level obs are promoted from the first modality that
    contains them (aligned by barcode index).
    """
    obs = mdata.obs.copy()
    keys_to_promote = [k for k in (batch_key, label_key) if k and k not in obs.columns]
    for key in keys_to_promote:
        for mod_adata in mdata.mod.values():
            if key in mod_adata.obs.columns:
                obs[key] = mod_adata.obs[key].reindex(obs.index)
                logger.info(
                    "Promoted obs key '%s' from modality to top-level obs.", key
                )
                break
        else:
            logger.warning(
                "Key '%s' not found in any modality obs; it will be absent.", key
            )
    return ad.AnnData(obs=obs)


def evaluate_single_run(
    output_dir: str,
    dataset_path: str,
    batch_key: str | None,
    label_key: str | None,
) -> dict:
    """Run per-job scIB evaluation after a model container completes.

    Loads the dataset, attaches embeddings from output_dir/embeddings.h5, runs
    scib-metrics, and writes results to output_dir/metrics.json.
    Returns the metrics dict (empty if embeddings are missing).
    """
    embeddings_path = os.path.join(output_dir, "embeddings.h5")
    if not os.path.exists(embeddings_path):
        logger.warning(
            "No embeddings.h5 in %s; skipping per-job evaluation.", output_dir
        )
        return {}

    embedding_only = False
    if dataset_path.endswith(".h5mu"):
        try:
            import mudata as md

            mdata = md.read_h5mu(dataset_path)
            adata = _mudata_to_evaluation_anndata(
                mdata, batch_key=batch_key, label_key=label_key
            )
            embedding_only = True
        except Exception as exc:
            logger.warning(
                "Failed to load/convert MuData from %s: %s", dataset_path, exc
            )
            return {}
    elif dataset_path.endswith(".h5ad"):
        adata = ad.read_h5ad(dataset_path)
        embedding_only = adata.n_vars == 0
    else:
        logger.warning("Unsupported dataset format for evaluation: %s", dataset_path)
        return {}

    with h5py.File(embeddings_path, "r") as f:
        latent = f["latent"][:]

    if latent.shape[0] != adata.n_obs:
        logger.warning(
            "Embedding row count (%d) != dataset obs count (%d); skipping evaluation.",
            latent.shape[0],
            adata.n_obs,
        )
        return {}

    latent_key = "X_model"
    adata.obsm[latent_key] = latent

    effective_batch_key = (
        batch_key if (batch_key and batch_key in adata.obs.columns) else None
    )
    if not effective_batch_key:
        logger.warning(
            "batch_key '%s' not found in obs; batch-correction metrics will be disabled.",
            batch_key,
        )
        # scib-metrics requires a batch_key argument even when all batch metrics
        # are disabled. A constant internal column satisfies API validation without
        # creating artificial batch-correction scores.
        adata.obs["_batch_unavailable"] = "batch_unavailable"

    effective_label_key = (
        label_key if (label_key and label_key in adata.obs.columns) else None
    )
    if label_key and not effective_label_key:
        logger.warning(
            "label_key '%s' not found in obs; supervised metrics will be skipped.",
            label_key,
        )

    have_labels = bool(effective_label_key)
    have_batch = bool(effective_batch_key)
    # pcr_comparison requires a valid expression matrix; skip when only obs metadata is available.
    have_matrix = not embedding_only

    bm = Benchmarker(
        adata,
        batch_key=effective_batch_key or "_batch_unavailable",
        label_key=effective_label_key,
        embedding_obsm_keys=[latent_key],
        bio_conservation_metrics=BioConservation(
            isolated_labels=have_labels,
            nmi_ari_cluster_labels_leiden=have_labels,
            nmi_ari_cluster_labels_kmeans=have_labels,
            silhouette_label=have_labels,
            clisi_knn=have_labels,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=have_batch,
            ilisi_knn=have_batch,
            kbet_per_label=have_batch and have_labels,
            graph_connectivity=have_batch,
            pcr_comparison=have_batch and have_matrix,
        ),
    )

    bm.benchmark()
    results_df = bm.get_results(min_max_scale=False)
    evaluation_metrics = (
        sanitize_nan_inf(results_df.to_dict("dict")) if not results_df.empty else {}
    )

    out_path = os.path.join(output_dir, "metrics.json")
    existing_metrics = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                existing_metrics = loaded
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not merge existing metrics from %s: %s", out_path, exc
            )

    merged_metrics = sanitize_nan_inf(
        {**existing_metrics, "evaluation": evaluation_metrics}
    )
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(merged_metrics, fp, indent=4)
    logger.info("Per-job evaluation metrics merged into %s", out_path)
    return merged_metrics


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
