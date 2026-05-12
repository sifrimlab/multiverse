"""TotalVI container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import random

import numpy as np
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
    config = build_model_config("totalvi", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed

    mdata = load_input_mudata()
    adata = anndata_concatenate(
        [mdata[m] for m in mdata.mod.keys()],
        list(mdata.mod.keys()),
    )

    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        batch_key="batch",
    )

    logger.info("Running TotalVI")
    model = scvi.model.TOTALVI(adata)
    model.train()

    save_embeddings(model.get_latent_representation(), OUTPUT_DIR)
    logger.info("TotalVI run complete.")


if __name__ == "__main__":
    main()
