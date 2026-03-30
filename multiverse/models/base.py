import os
import numpy as np
import scanpy as sc
import h5py
import matplotlib.pyplot as plt
from ..config import load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Other classes will inherit initial attributes of this class (config_file, dataset, dataset_name, ...)
    List of functions in each model is same as ModelFactory but how it works is different for each model
    """

    def __init__(
        self,
        dataset,
        dataset_name: str,
        model_name: str = "",
        config_path: str = "./config.json",
        is_gridsearch=False,
    ):
        self.config_dict = load_config(config_path=config_path)
        self.model_params = self.config_dict.get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = os.path.join(
            self.config_dict["output_dir"],
            self.dataset_name,
            self.model_name,
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

    def update_parameters(self, **kwargs):
        """
        Updates the model parameters.
        Args:
            **kwargs: Keyword arguments with parameter names and their new values.
                     Example: update_parameters(n_factors=10, n_iteration=500)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle invalid parameter names if necessary
                logger.warning(f"Invalid parameter name '{key}'")

    def train(self):
        logger.info("Training the model.")

    def save_latent(self):
        """
        Saves only the latent embedding (matrix) to an HDF5 file.
        """
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")

        latent = self.dataset.obsm[self.latent_key]
        if not isinstance(latent, np.ndarray):
            latent = latent.to_numpy()  # handle sparse or dataframe

        try:
            logger.info("Saving latent embedding matrix")
            with h5py.File(self.latent_filepath, "w") as f:
                f.create_dataset("latent", data=latent)
            logger.info(f"Latent embedding saved to {self.latent_filepath}")
        except IOError as e:
            logger.error(f"Could not write latent file to {self.latent_filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving latent data: {e}")
            raise

    def umap(self):
        """Generate UMAP visualization using Cobolt embeddings for all modalities."""
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

    def evaluate_model(self):
        logger.info("Evaluating the model.")
