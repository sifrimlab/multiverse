"""Cobolt container entrypoint. Reads /input/data.h5mu, writes /output/embeddings.h5."""
import random

import numpy as np
import scipy.sparse
import torch
from cobolt.model import Cobolt
from cobolt.utils import MultiomicDataset, SingleData

from mvr_worker import (
    OUTPUT_DIR,
    build_model_config,
    get_logger,
    load_input_mudata,
    load_job_spec,
    resolve_device,
    save_embeddings,
    setup_container_logging,
)

logger = get_logger(__name__)


def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    job_spec = load_job_spec()
    config = build_model_config("cobolt", job_spec, OUTPUT_DIR)

    seed = config.get("seed") or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    params = config["model"].get("cobolt", {})
    latent_dim = params.get("latent_dimensions", 20)
    lr = params.get("learning_rate", 0.001)
    num_epochs = params.get("num_epochs", 200)
    device = resolve_device(params.get("device", "cpu"))

    mdata = load_input_mudata()
    dataset_name = job_spec.get("dataset_name") or "dataset"
    modalities = list(mdata.mod.keys())

    single_data_list = []
    for mod in modalities:
        adata = mdata[mod]
        X = adata.X if scipy.sparse.issparse(adata.X) else scipy.sparse.csr_matrix(adata.X)
        single_data_list.append(
            SingleData(
                feature_name=mod,
                dataset_name=dataset_name,
                feature=adata.var_names.to_numpy(),
                count=X,
                barcode=adata.obs_names.to_numpy(),
            )
        )

    multiomic = MultiomicDataset.from_singledata(*single_data_list)

    logger.info(f"Running Cobolt with latent_dim={latent_dim}, device={device}")
    model = Cobolt(dataset=multiomic, n_latent=latent_dim, lr=lr, device=device)
    model.train(num_epochs=num_epochs)

    all_latent = model.get_all_latent()[0]
    comb_idx = multiomic.get_comb_idx([True] * len(multiomic.omic))
    latent = all_latent[[comb_idx]].squeeze(0)

    save_embeddings(latent, OUTPUT_DIR)
    logger.info("Cobolt run complete.")


if __name__ == "__main__":
    main()
