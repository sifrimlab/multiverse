"""MultiVI container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
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
    save_embeddings,
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

    params = config["model"].get("multivi", {})

    mdata = load_input_mudata()
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

    save_embeddings(model.get_latent_representation(), OUTPUT_DIR)
    logger.info("MultiVI run complete.")


if __name__ == "__main__":
    main()
