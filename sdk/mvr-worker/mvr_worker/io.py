from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Union

import anndata as ad
import h5py
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc

from .logging import get_logger, setup_logging

logger = get_logger(__name__)

OUTPUT_DIR = os.environ.get("MVR_OUTPUT_DIR", "/output")
INPUT_DATA_PATH = os.environ.get("MVR_INPUT_DATA_PATH", "/input/data.h5mu")
JOB_SPEC_PATH = os.environ.get(
    "MVR_JOB_SPEC_PATH", os.path.join(OUTPUT_DIR, "job_spec.json")
)


def setup_container_logging(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)


def load_job_spec(path: str = JOB_SPEC_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_input_mudata(path: str = INPUT_DATA_PATH) -> md.MuData:
    return md.read_h5mu(path)


def build_model_config(
    model_name: str,
    job_spec: Dict[str, Any],
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, Any]:
    params = job_spec.get("hyperparameters", {})
    scoped = {model_name: params.get(model_name, params)}
    return {
        "output_dir": output_dir,
        "seed": job_spec.get("seed"),
        "model": scoped,
    }


def save_embeddings(latent: np.ndarray, output_dir: str = OUTPUT_DIR) -> str:
    """Atomically write latent matrix to <output_dir>/embeddings.h5 (key: 'latent')."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "embeddings.h5")
    tmp = path + ".tmp"
    arr = np.asarray(latent, dtype=np.float32)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("latent", data=arr)
    os.rename(tmp, path)
    logger.info(f"Embeddings saved to {path} — shape {arr.shape}")
    return path


def save_umap(
    latent: np.ndarray,
    obs: "pd.DataFrame",
    output_dir: str = OUTPUT_DIR,
    color_key: str = "cell_type",
    random_state: int = 42,
) -> "str | None":
    """Generate UMAP from latent embeddings and save to <output_dir>/umap.png.

    Non-fatal: returns None and logs a warning on failure so the container
    run is not aborted due to a visualization error.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc

        tmp = ad.AnnData(obs=obs.copy())
        tmp.obsm["X_latent"] = np.asarray(latent, dtype=np.float32)

        sc.pp.neighbors(tmp, use_rep="X_latent", random_state=random_state)
        sc.tl.umap(tmp, random_state=random_state)

        effective_color = color_key if color_key and color_key in tmp.obs else None
        sc.pl.umap(tmp, color=effective_color, show=False)

        path = os.path.join(output_dir, "umap.png")
        fd, tmp_path = tempfile.mkstemp(
            prefix=".umap-",
            suffix=".png",
            dir=output_dir,
        )
        os.close(fd)
        try:
            plt.savefig(tmp_path, format="png", bbox_inches="tight", dpi=150)
            os.replace(tmp_path, path)
        finally:
            plt.close()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        logger.info(f"UMAP saved to {path}")
        return path
    except Exception as exc:
        logger.warning(f"UMAP generation failed (non-fatal): {exc}")
        return None


def anndata_concatenate(
    mdata: md.MuData = None,
    adata_list: list = None,
    selected_modalities: list = None,
    obs: pd.DataFrame = None,
    cell_type_key: str = "cell_type",
    batch_key: str = "batch",
) -> ad.AnnData:
    """
    Fuse modalities and concatenate along variables axis.
    This is a common preprocessing step for models that require a single AnnData input.
    It also ensures that cell type and batch annotations are preserved in the concatenated object.
    Note: this function assumes that the same cells are present across all modalities and that
    cell type and batch annotations are consistent across modalities.
    :param mdata: MuData object containing multiple modalities
    :param adata_list: List of AnnData objects to concatenate (if None, will be constructed from mdata.mod using selected_modalities)
    :param selected_modalities: List of modalities to include in the concatenated object
    :param cell_type_key: Key in .obs that contains cell type annotations
    :param batch_key: Key in .obs that contains batch annotations
    :return: Concatenated AnnData object with modalities fused along variables axis
     and cell type and batch annotations preserved in .obs.

    """
    if adata_list is not None and mdata is not None:
        raise ValueError("Provide either adata_list or mdata, not both.")
    if adata_list is None and mdata is None:
        raise ValueError("Either adata_list or mdata must be provided.")
    if selected_modalities is None and mdata is None:
        raise ValueError(
            "selected_modalities has to be provided when mdata is not provided."
        )
    if obs is None and mdata is None:
        raise ValueError("obs has to be provided when mdata is not provided.")
    if obs is None:
        obs = mdata.obs
    if selected_modalities is None and mdata is not None:
        selected_modalities = list(mdata.mod.keys())

    if adata_list is None:
        list_mod_adata = [
            mdata[m] for m in selected_modalities if m in mdata.mod.keys()
        ]
    else:
        list_mod_adata = adata_list
    adata_concat = ad.concat(
        list_mod_adata,
        axis="var",
        label=cell_type_key,
        merge="unique",
        uns_merge="unique",
    )
    adata_concat.obs[batch_key] = obs[batch_key]
    adata_concat.obs[cell_type_key] = obs[cell_type_key]
    adata_concat.obs["modality"] = np.zeros(adata_concat.n_obs, dtype=int)
    return adata_concat


def load_config(config_path: Union[str, dict] = "./config.json"):
    """Load the configuration from a JSON file or return an in-memory dict unchanged.

    Args:
        config_path: Path to the JSON configuration file, or a configuration dict.

    Returns:
        dict: Dictionary of hyperparameters and settings.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
        Exception: For any other unexpected errors during file loading.
    """
    if isinstance(config_path, dict):
        return config_path

    try:
        logger.info("Loading .json file")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        logger.info("Information from json file loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading the configuration file: {e}"
        )
        raise
    return config


def resolve_preprocess_params(job_spec, modalities, defaults):
    """Resolve effective preprocessing parameters for a run (issue #22).

    ``defaults`` are the model's built-in preprocessing parameters (the values
    previously hard-coded in each ``run.py``); they remain authoritative so an
    absent ``preprocessing`` block reproduces the legacy behaviour exactly.
    Any non-null key in ``job_spec["preprocessing"]`` (the per-run override
    resolved from the run manifest / GUI) takes precedence.

    ``scale`` accepts either a per-modality mapping or a single bool applied to
    every modality, in both ``defaults`` and the override.
    """
    resolved = dict(defaults or {})
    # Normalise a bool ``scale`` default into a per-modality mapping.
    if "scale" in resolved and not isinstance(resolved["scale"], dict):
        resolved["scale"] = {mod: bool(resolved["scale"]) for mod in modalities}

    overrides = (job_spec or {}).get("preprocessing") or {}
    for key, val in dict(overrides).items():
        if val is None:
            continue
        if key == "scale":
            if isinstance(val, dict):
                merged = dict(resolved.get("scale") or {})
                merged.update({m: bool(v) for m, v in val.items()})
                resolved["scale"] = merged
            else:
                resolved["scale"] = {mod: bool(val) for mod in modalities}
        else:
            resolved[key] = val
    return resolved


def preprocess_mudata(
    mdata,
    preprocess_params,
    cell_type_key="cell_type",
    batch_key="batch",
):
    # TODO: make modality specific preprocessing configurable, e.g. log normalization only for RNA, not ADT, etc.
    """Preprocess MuData while keeping all modalities aligned."""

    modalities = list(mdata.mod.keys())

    # ------------------------------------------------------------------
    # Step 1: Filter genes independently
    # ------------------------------------------------------------------
    for modality in modalities:
        logger.info(f"Preprocessing modality: {modality}")
        sc.pp.filter_genes(mdata.mod[modality], min_cells=1)

    # ------------------------------------------------------------------
    # Step 2: Find cells with counts in ALL modalities
    # ------------------------------------------------------------------
    common_cells = None

    for modality in modalities:
        adata = mdata.mod[modality]

        counts_per_cell = np.asarray(adata.X.sum(axis=1)).ravel()
        valid_cells = adata.obs_names[counts_per_cell > 0]

        if common_cells is None:
            common_cells = set(valid_cells)
        else:
            common_cells &= set(valid_cells)

    common_cells = sorted(common_cells)

    # ------------------------------------------------------------------
    # Step 3: Subset ALL modalities to identical cells
    # ------------------------------------------------------------------
    for modality in modalities:
        mdata.mod[modality] = mdata.mod[modality][common_cells].copy()

    mdata.update()

    # ------------------------------------------------------------------
    # Step 4: Modality-specific preprocessing
    # ------------------------------------------------------------------
    for modality in modalities:
        adata = mdata.mod[modality]

        if modality.lower() == "rna":
            target_sum = preprocess_params.get("normalization_target_sum", None)

            if target_sum is not None:
                sc.pp.normalize_total(
                    adata,
                    target_sum=target_sum,
                )

            if preprocess_params.get(
                "log_normalization",
                False,
            ):
                sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # Step 5: HVG selection
    # ------------------------------------------------------------------
    n_top_genes = preprocess_params.get("n_top_genes")

    if n_top_genes is not None:
        for modality in modalities:
            adata = mdata.mod[modality]
            if adata.n_vars <= n_top_genes:
                logger.warning(
                    f"Warning: Modality '{modality}' has only {adata.n_vars} features, which is less than or equal to n_top_genes={n_top_genes}. Skipping HVG selection for this modality."
                )

            hvg_flavor = (
                "seurat"
                if (
                    modality.lower() == "rna"
                    and preprocess_params.get("log_normalization", False)
                )
                else "seurat_v3"
            )

            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=min(
                    n_top_genes,
                    adata.n_vars,
                ),
                flavor=hvg_flavor,
            )

            mdata.mod[modality] = adata[
                :,
                adata.var["highly_variable"],
            ].copy()

        mdata.update()

    """ 
        # --------------------------------------------------------------
        # Step 6: Remove cells that became empty after HVG filtering
        # --------------------------------------------------------------
        common_cells = None

        for modality in modalities:
            adata = mdata.mod[modality]

            counts_per_cell = np.asarray(adata.X.sum(axis=1)).ravel()

            valid_cells = adata.obs_names[counts_per_cell > 0]

            if common_cells is None:
                common_cells = set(valid_cells)
            else:
                common_cells &= set(valid_cells)

        common_cells = sorted(common_cells)

        for modality in modalities:
            mdata.mod[modality] = mdata.mod[modality][common_cells].copy()

        mdata.update()"""

    # ------------------------------------------------------------------
    # Step 7: Scaling
    # ------------------------------------------------------------------
    modality_scaling = preprocess_params.get("scale", {})
    for modality in modalities:
        if modality_scaling.get(modality, False):
            sc.pp.scale(mdata.mod[modality])

        mdata.update()

    # ------------------------------------------------------------------
    # Step 8: Synchronize metadata
    # ------------------------------------------------------------------
    ref_obs = mdata.mod[modalities[0]].obs

    mdata.obs[cell_type_key] = (
        ref_obs[cell_type_key] if cell_type_key in ref_obs.columns else "unknown"
    )

    mdata.obs[batch_key] = (
        ref_obs[batch_key] if batch_key in ref_obs.columns else "unknown"
    )
    mdata.var_names_make_unique()
    logger.info("Preprocessing completed.")
    return mdata
