# Model Container Contract

This reference defines the I/O contract that every model container must honour to be runnable under the mvexp orchestrator. It is the canonical specification; both the built-in models under `store/models/` and any third-party model image must conform to it.

## Filesystem Boundary

| Path | Direction | Required | Description |
|---|---|---|---|
| `/input/data.h5mu` | read | yes | Dataset materialized by the orchestrator. For single-modality datasets the file still uses `MuData` with a single modality. |
| `/output/job_spec.json` | read | yes | Per-job runtime instruction written by the orchestrator before the container starts. |
| `/output/embeddings.h5` | write | yes | HDF5 file containing exactly one top-level dataset named `latent`. |
| `/output/metrics.json` | write | yes | JSON object with model-level metrics and (optionally) training history. |
| `/output/umap.png` | write | yes | UMAP scatter of the latent space rendered by the container. |
| `/output/model.log` | write | yes | Structured log (the `mvr_worker.setup_container_logging` helper writes this for you). |

Model code must read no other host paths and write to no other host paths.

## `job_spec.json` Schema

| Field | Type | Description |
|---|---|---|
| `seed` | integer | Deterministic seed. The container must apply this to `random`, `numpy`, and (where applicable) `torch` before training. |
| `dataset_id` | integer \| null | Registry primary key. May be null when running offline. |
| `dataset_name` | string | Dataset display name or slug. |
| `model_name` | string | Model slug (lowercase) — `pca`, `mofa`, `multivi`, `mowgli`, `cobolt`, `totalvi`. |
| `hyperparameters` | object | Free-form key-value parameters selected in the GUI or manifest. Must conform to the model's hyperparameter JSON schema. |
| `run_settings` | object | Experiment-level settings: `experiment_name`, optional `tags`. |

Example:

```json
{
  "seed": 42,
  "dataset_id": 7,
  "dataset_name": "pbmc10k",
  "model_name": "pca",
  "hyperparameters": {"n_components": 20, "device": "cpu"},
  "run_settings": {"experiment_name": "pbmc-baselines"}
}
```

## `embeddings.h5` Format

```text
/
└── latent     shape: (n_cells, n_dim), dtype: float32 or float64
```

The number of rows must equal the number of cells in the input. Embedding row ordering must match the `obs` ordering of `/input/data.h5mu`.

Compliance check:

```python
import h5py

with h5py.File("/output/embeddings.h5", "r") as f:
    latent = f["latent"][:]

assert latent.ndim == 2 and latent.shape[0] > 0 and latent.shape[1] > 0
```

## `metrics.json` Format

```json
{
  "model_metrics": {"reconstruction_loss": 0.12, "elbo": -1234.5},
  "history": {
    "epoch": [1, 2, 3],
    "train_loss": [10.0, 5.0, 2.0]
  }
}
```

The `model_metrics` map should contain finite scalars only. `NaN` and `±Inf` values are sanitised by the tracking layer but degrade comparability across runs. `history` is optional and consumed by MLflow as a per-epoch metric stream when present.

## Container Authoring with `mvr-worker`

The container-side SDK at `sdk/mvr-worker/` provides every helper needed to honour the contract. The expected import surface is:

```python
from mvr_worker import (
    OUTPUT_DIR,              # "/output"
    load_input_mudata,       # reads /input/data.h5mu
    load_job_spec,           # parses /output/job_spec.json
    build_model_config,      # resolves hyperparameters with sensible defaults
    save_embeddings,         # writes /output/embeddings.h5 with the latent matrix
    save_umap,               # writes /output/umap.png
    anndata_concatenate,     # multimodal feature concatenation
    setup_container_logging, # configures /output/model.log
    get_logger,              # named logger
    EpochLogger,             # context manager streaming epoch metrics to MLflow + JSONL
    resolve_device,          # CPU/CUDA selection
)
```

Reference implementation (PCA, paraphrased from `store/models/pca/container/run.py`):

```python
import random, numpy as np, scanpy as sc
from mvr_worker import (
    OUTPUT_DIR, anndata_concatenate, build_model_config, get_logger,
    load_input_mudata, load_job_spec, save_embeddings, save_umap,
    setup_container_logging,
)

def main() -> None:
    setup_container_logging(OUTPUT_DIR)
    spec = load_job_spec()
    cfg = build_model_config("pca", spec, OUTPUT_DIR)

    seed = cfg.get("seed") or 42
    random.seed(seed); np.random.seed(seed)

    mdata = load_input_mudata()
    adata = anndata_concatenate([mdata[m] for m in mdata.mod], list(mdata.mod))

    sc.pp.pca(adata, n_comps=cfg["model"]["pca"].get("n_components", 50))
    save_embeddings(adata.obsm["X_pca"], OUTPUT_DIR)
    save_umap(adata.obsm["X_pca"], adata.obs, OUTPUT_DIR)
```

Every built-in model image follows this skeleton.

## Build Pattern

Container Dockerfiles use `mambaorg/micromamba` and install the SDK from the build context:

```dockerfile
FROM mambaorg/micromamba:2.3.0
USER root
WORKDIR /app

COPY store/models/<slug>/container/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy
ENV PATH=/opt/conda/envs/<env-name>/bin:$PATH

COPY sdk/mvr-worker/ /tmp/mvr-worker/
RUN pip install /tmp/mvr-worker/

COPY store/models/<slug>/container/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
```

The build context is the repository root so that `COPY sdk/mvr-worker/ ...` resolves correctly; see the `build:` block of `store/models/<slug>/model.yaml`.

## Determinism Rules

1. Apply `job_spec.json:seed` to `random`, `numpy`, and `torch` before any stochastic operation.
2. Avoid wall-clock-derived seeds, including UMAP defaults; honour `umap_random_state` from the hyperparameters when present.
3. Do not pin to GPU device 0 implicitly; use `resolve_device()` and honour the `device` hyperparameter.

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `embeddings.h5` missing on success path | Container exited before writing outputs. | Wrap I/O in a `try/finally`; flush before exit. |
| `latent` key missing in HDF5 | Wrong dataset name. | Use `save_embeddings()` from `mvr_worker`. |
| Row count mismatch with input | Filtering applied after `mdata.obs` was captured. | Filter the input once and reuse the same indexing. |
| `metrics.json` invalid JSON | Manual string concatenation. | Use `json.dump` and write scalars only. |
| Run unreproducible across hosts | Hidden state in CUDA kernels or library defaults. | Seed all RNGs; set `torch.use_deterministic_algorithms(True)` where supported. |
