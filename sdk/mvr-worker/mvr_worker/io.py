from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import anndata as ad
import h5py
import mudata as md
import numpy as np
import pandas as pd

from .logging import get_logger, setup_logging

logger = get_logger(__name__)

OUTPUT_DIR = os.environ.get("MVR_OUTPUT_DIR", "/output")
INPUT_DATA_PATH = os.environ.get("MVR_INPUT_DATA_PATH", "/input/data.h5mu")
JOB_SPEC_PATH = os.environ.get("MVR_JOB_SPEC_PATH", os.path.join(OUTPUT_DIR, "job_spec.json"))


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
    list_anndata: List[ad.AnnData],
    list_modality: List[str],
) -> ad.AnnData:
    """Fuse modalities and concatenate along variables axis.

    Reimplements multiverse/data_utils.py without the muon dependency —
    obs intersection is done with plain set arithmetic on mudata.MuData.
    """
    if len(list_modality) != len(list_anndata):
        raise ValueError("list_modality and list_anndata must have equal length.")

    data_dict: Dict[str, ad.AnnData] = {}
    for mod, adata in zip(list_modality, list_anndata):
        adata = adata.copy()
        if hasattr(adata.X, "toarray"):
            adata.X = np.array(adata.X.toarray())
        data_dict[mod] = adata

    mdata = md.MuData(data_dict)

    # Intersect observations across modalities (replaces muon.pp.intersect_obs)
    common_idx = sorted(
        set.intersection(*[set(mdata.mod[m].obs_names) for m in mdata.mod])
    )
    for key in list(mdata.mod.keys()):
        mdata.mod[key] = mdata.mod[key][common_idx].copy()
    mdata.update()

    # Propagate cell_type from rna modality when available
    if "rna" in mdata.mod and "cell_type" in mdata["rna"].obs.columns:
        mdata.obs["cell_type"] = mdata["rna"].obs["cell_type"]
    else:
        logger.warning("No 'cell_type' annotation found — supervised metrics unavailable.")
        mdata.obs["cell_type"] = "unknown"

    # Concatenate modalities along variable axis
    list_mod_adata = [mdata[m] for m in list_modality]
    adata_concat = ad.concat(
        list_mod_adata, axis="var", label="cell_type", merge="unique", uns_merge="unique"
    )
    adata_concat.obs["cell_type"] = mdata.obs["cell_type"]
    adata_concat.obs["modality"] = np.zeros(adata_concat.n_obs, dtype=int)
    return adata_concat
