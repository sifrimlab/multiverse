"""Mowgli container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import json
import os
import random

import mowgli
import numpy as np
import scanpy as sc
import torch

from mvr_worker import (
    OUTPUT_DIR,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    replay_history,
    resolve_device,
    save_embeddings,
    save_umap,
    setup_container_logging,
)

logger = get_logger(__name__)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("mowgli", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    params = config["model"].get("mowgli", {})
    latent_dim = params.get("latent_dimensions", 20)
    optimizer = params.get("optimizer", "adam")
    lr = params.get("learning_rate", 0.001)
    tol_inner = params.get("tol_inner", 1e-6)
    max_iter_inner = params.get("max_iter_inner", 500)
    device = resolve_device(params.get("device", "cpu"))

    mdata = load_input_mudata()
    dataset_name = job_spec.get("dataset_name") or "dataset"

    # Mowgli requires highly_variable annotation on every modality var.
    for mod in list(mdata.mod.keys()):
        adata = mdata[mod]
        if "highly_variable" not in adata.var.columns:
            if mod == "rna":
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars))
            else:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=min(20000, adata.n_vars))

    logger.info(f"Running Mowgli with latent_dim={latent_dim}, device={device}")
    model = mowgli.models.MowgliModel(latent_dim=latent_dim)
    model.train(
        mdata,
        device=device,
        optim_name=optimizer,
        lr=lr,
        tol_inner=tol_inner,
        max_iter_inner=max_iter_inner,
    )

    latent = mdata.obsm["W_OT"]
    save_embeddings(latent, OUTPUT_DIR)
    save_umap(latent, mdata.obs, OUTPUT_DIR)

    raw_losses = list(getattr(model, "losses", []) or [])
    # Mowgli minimizes -OT loss internally; flip sign so the curve trends down to zero.
    ot_loss_series = [-float(v) for v in raw_losses]
    history = replay_history(
        {"ot_loss": ot_loss_series} if ot_loss_series else {},
        output_dir=OUTPUT_DIR,
        run_name=f"{dataset_name}-mowgli-{os.path.basename(OUTPUT_DIR)}",
    )

    payload: dict = {}
    if "ot_loss" in history:
        payload["ot_loss"] = history["ot_loss"][-1]
    if history:
        payload["history"] = history
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    logger.info("Mowgli run complete.")


if __name__ == "__main__":
    main()
