"""PCA container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import random

import numpy as np
import scanpy as sc

from mvr_worker import (
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    save_embeddings,
    save_umap,
    setup_container_logging,
)

logger = get_logger(__name__)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("pca", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)

    params = config["model"].get("pca", {})
    n_components = params.get("n_components", 50)

    mdata = load_input_mudata()
    adata = anndata_concatenate(
        [mdata[m] for m in mdata.mod.keys()],
        list(mdata.mod.keys()),
    )

    logger.info(f"Running PCA with n_components={n_components}")
    if "highly_variable" in adata.var:
        sc.pp.pca(adata, n_comps=n_components, use_highly_variable=True)
    else:
        sc.pp.pca(adata, n_comps=n_components, use_highly_variable=False)

    latent = adata.obsm["X_pca"]
    save_embeddings(latent, OUTPUT_DIR)
    save_umap(latent, adata.obs, OUTPUT_DIR)
    logger.info("PCA run complete.")


if __name__ == "__main__":
    main()
