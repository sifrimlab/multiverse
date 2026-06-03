import json
import os
from typing import Any, Dict, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from .epoch_logger import sanitize_nan_inf
from .io import load_config
from .logging import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """The base class for all multimodal integration model wrappers.

    Standardizes the lifecycle of a model run, providing methods for training,
    saving latent embeddings, and generating visualizations.

    Attributes:
        dataset (ad.AnnData or md.MuData): The input dataset.
        dataset_name (str): A descriptive name for the dataset.
        model_name (str): The name of the model being executed.
        output_dir (str): Base directory for model results.
        latent_filepath (str): Path to save latent embeddings.
        umap_filename (str): Path to save the UMAP plot.
        metrics_filepath (str): Path to save model-specific metrics.
    """

    def __init__(
        self,
        dataset,
        dataset_name: str,
        model_name: str = "",
        config_path: Union[str, dict] = "./config.json",
        is_gridsearch: bool = False,
    ):
        """Initializes the ModelFactory base class.

        Args:
            dataset (Union[ad.AnnData, md.MuData]): The dataset object.
            dataset_name (str): Name of the dataset.
            model_name (str): Name of the model. Defaults to "".
            config_path (Union[str, dict]): Path to the configuration file or the configuration dictionary.
                Defaults to "./config.json".
            is_gridsearch (bool): Flag indicating if this is a grid search run. Defaults to False.
        """
        if isinstance(config_path, dict):
            self.config_dict = config_path
        else:
            self.config_dict = load_config(config_path)
        self.model_params = self.config_dict.get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = os.path.join(
            self.config_dict["output_dir"],
        )

        # Embeddings of the latent space
        self.latent_filepath = os.path.join(
            self.output_dir,
            "embeddings.h5",
        )
        self.umap_filename = os.path.join(
            self.output_dir,
            "umap.png",
        )
        self.metrics_filepath = os.path.join(
            self.output_dir,
            "metrics.json",
        )
        self.is_grid_search = is_gridsearch  # Flag for grid search runs
        os.makedirs(self.output_dir, exist_ok=True)
        self.latent_key = f"X_{self.model_name}"
        if self.model_name in self.model_params:
            model_specific_params = self.model_params.get(self.model_name)
            self.umap_color_type = model_specific_params.get("umap_color_type")

        if self.model_name in self.model_params:
            model_specific_params = self.model_params.get(self.model_name)
            self.umap_color_type = model_specific_params.get("umap_color_type")

    def train(self):
        """Abstract method for training the model. Subclasses must implement this."""
        logger.info("Training the model.")

    def save_latent(self):
        """Saves the calculated latent representation of the data to an HDF5 file.

        The latent representation is expected to be stored in `self.dataset.obsm`
        under the key `self.latent_key`.

        Raises:
            ValueError: If `latent_filepath` is not set.
            IOError: If there is an issue writing the HDF5 file.
        """
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")

        latent = self.dataset.obsm[self.latent_key]
        if not isinstance(latent, np.ndarray):
            latent = latent.to_numpy()  # handle sparse or dataframe

        tmp_filepath = f"{self.latent_filepath}.tmp"
        try:
            logger.info(
                f"Saving latent embedding matrix to temporary file: {tmp_filepath}"
            )
            with h5py.File(tmp_filepath, "w") as f:
                f.create_dataset("latent", data=latent)

            # Atomic rename
            os.rename(tmp_filepath, self.latent_filepath)
            logger.info(f"Latent embedding saved to {self.latent_filepath}")
        except IOError as e:
            logger.error(f"Could not write latent file to {tmp_filepath}: {e}")
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving latent data: {e}")
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            raise

    def umap(self):
        """Generates a UMAP visualization using the model's latent embeddings.

        The resulting plot is saved to `self.umap_filename`.

        Raises:
            ValueError: If `umap_filename` is not set.
            Exception: If an error occurs during UMAP generation or plotting.
        """
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")

        logger.info(
            f"Generating UMAP with {self.model_name} embeddings for all modalities"
        )
        try:
            sc.pp.neighbors(
                self.dataset,
                use_rep=self.latent_key,
                random_state=self.umap_random_state,
            )

            sc.tl.umap(self.dataset, random_state=self.umap_random_state)

            self.dataset.obsm[f"X_{self.model_name}_umap"] = self.dataset.obsm[
                "X_umap"
            ].copy()

            if self.umap_color_type in self.dataset.obs:
                sc.pl.umap(self.dataset, color=self.umap_color_type, show=False)
            else:
                logger.warning(
                    f"UMAP color key '{self.umap_color_type}' not found in .obs. Plotting without color."
                )
                sc.pl.umap(self.dataset, show=False)

            plt.savefig(self.umap_filename, bbox_inches="tight")
            plt.close()

            logger.info(
                f"UMAP plot for {self.model_name} {self.dataset_name} saved as {self.umap_filename}"
            )
        except Exception as e:
            logger.error(f"An error occurred during UMAP generation: {e}")
            raise

    def write_metrics(
        self,
        metrics: Dict[str, Any],
        history: Dict[str, Any] | None = None,
    ) -> None:
        """Persist final scalars and optional per-epoch history to metrics.json.

        Writes both the nested model path and the flat container contract path
        at ``<output_dir>/metrics.json`` so the orchestrator and MLflow can find them.
        """
        payload: Dict[str, Any] = dict(metrics)
        if history:
            import math

            cleaned_history = {}
            for key, values in history.items():
                if not values:
                    continue
                series = []
                for value in values:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(numeric):
                        series.append(numeric)
                if series:
                    cleaned_history[str(key)] = series
            if cleaned_history:
                payload["history"] = cleaned_history

        payload = sanitize_nan_inf(payload)

        paths = {
            self.metrics_filepath,
            os.path.join(self.config_dict["output_dir"], "metrics.json"),
        }
        last_error: Exception | None = None
        for path in paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(payload, fp, indent=4)
            except IOError as exc:
                last_error = exc
                logger.error("Could not write metrics file to %s: %s", path, exc)
        if last_error is not None and not any(os.path.exists(p) for p in paths):
            raise last_error
        logger.info("Metrics saved to %s", self.metrics_filepath)

    def evaluate_model(self):
        """Abstract method for evaluating the model. Subclasses must implement this."""
        logger.info("Evaluating the model.")
