from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import anndata as ad
import h5py
import mudata as md
import numpy as np

from .logging import get_logger, setup_logging

logger = get_logger(__name__)

INPUT_DATA_PATH = "/input/data.h5mu"
OUTPUT_DIR = "/output"
JOB_SPEC_PATH = "/output/job_spec.json"


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
