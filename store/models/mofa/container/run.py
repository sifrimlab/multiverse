"""MOFA container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import random

import muon as mu
import numpy as np
import scanpy as sc

from mvr_worker import (
    OUTPUT_DIR,
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
    config = build_model_config("mofa", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)

    params = config["model"].get("mofa", {})
    n_factors = params.get("n_factors", 20)
    n_iterations = params.get("n_iterations", 5000)
    device = params.get("device", "cpu")
    gpu_mode = device != "cpu"

    mdata = load_input_mudata()

    # Select highly variable features per modality to keep memory manageable.
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
        mdata.mod[mod] = adata[:, adata.var["highly_variable"]].copy()
    mdata.update()

    logger.info(f"Running MOFA+ with n_factors={n_factors}, gpu_mode={gpu_mode}")
    mu.tl.mofa(mdata, n_factors=n_factors, gpu_mode=gpu_mode)

    latent = mdata.obsm["X_mofa"]
    save_embeddings(latent, OUTPUT_DIR)
    logger.info("MOFA run complete.")


if __name__ == "__main__":
    main()
