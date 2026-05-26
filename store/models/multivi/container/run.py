"""MultiVI container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import json
import os
import random

import numpy as np
import pandas as pd
import scvi

from mvr_worker import (
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    replay_history,
    save_embeddings,
    save_umap,
    setup_container_logging,
)

logger = get_logger(__name__)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("multivi", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed

    mdata = load_input_mudata()
    dataset_name = job_spec.get("dataset_name") or "dataset"
    adata = anndata_concatenate(
        [mdata[m] for m in mdata.mod.keys()],
        list(mdata.mod.keys()),
    )

    if "feature_types" not in adata.var:
        raise ValueError("MultiVI requires 'feature_types' in adata.var.")

    adata = adata[:, adata.var["feature_types"].argsort()].copy()

    if "Protein Expression" in adata.var["feature_types"].unique():
        protein_idx = adata.var["feature_types"] == "protein expression"
        prot_expr = pd.DataFrame(
            adata.X[:, protein_idx],
            index=adata.obs_names,
            columns=adata.var_names[protein_idx],
        )
        adata.obsm["protein_expression"] = prot_expr
        scvi.model.MULTIVI.setup_anndata(adata, protein_expression_obsm_key="protein_expression")
    else:
        scvi.model.MULTIVI.setup_anndata(adata, protein_expression_obsm_key=None)

    n_genes = (adata.var["feature_types"] == "Gene Expression").sum()
    n_regions = (adata.var["feature_types"] == "Peaks").sum()

    logger.info(f"Running MultiVI with n_genes={n_genes}, n_regions={n_regions}")
    model = scvi.model.MULTIVI(adata, n_genes=n_genes, n_regions=n_regions)
    model.train()

    latent = model.get_latent_representation()
    save_embeddings(latent, OUTPUT_DIR)
    save_umap(latent, adata.obs, OUTPUT_DIR)

    raw_history = getattr(model, "history", None) or {}
    if hasattr(raw_history, "keys"):
        raw_history = {k: raw_history[k] for k in raw_history.keys()}
    history = replay_history(
        raw_history,
        output_dir=OUTPUT_DIR,
        run_name=f"{dataset_name}-multivi-{os.path.basename(OUTPUT_DIR)}",
    )

    payload: dict = {}
    for key in ("elbo_train", "reconstruction_loss_train"):
        if key in history:
            payload[key] = history[key][-1]
    if history:
        payload["history"] = history
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    logger.info("MultiVI run complete.")


if __name__ == "__main__":
    main()
