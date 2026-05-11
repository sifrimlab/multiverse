# Contributing Guide

This document follows the [Diátaxis framework](https://diataxis.fr/): Tutorials (learning-oriented),
How-to Guides (task-oriented), Explanation (understanding-oriented), and Reference (information-oriented).

---

## Model Onboarding

### Tutorial: Add Your First Model in 15 Minutes

This tutorial walks you through adding a minimal model from scratch. By the end you will have
a working containerized model that appears in the GUI, runs via `make benchmark`, and produces
a `metrics.json` that feeds the end-of-run summary.

We will add a simple **RandomEmbedding** model (useful as a sanity-check baseline).

**Prerequisites:** Docker installed and running, `make install` already done.

#### Step 1 — Create the model directory

```bash
mkdir -p store/models/random_embedding/container
```

#### Step 2 — Write the model Python file

Create `multiverse/models/random_embedding.py`:

```python
import numpy as np
from .base import ModelFactory
from .runtime_io import build_model_config, load_input_mudata, load_job_spec, setup_container_logging
from ..logging_utils import get_logger

logger = get_logger(__name__)


class RandomEmbeddingModel(ModelFactory):
    def __init__(self, dataset, dataset_name, config_path, is_gridsearch=False):
        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="random_embedding", is_gridsearch=is_gridsearch)
        params = self.model_params.get("random_embedding", {})
        self.n_dims = params.get("n_dims", 32)
        self.umap_random_state = params.get("umap_random_state", 42)
        self.umap_color_type = params.get("umap_color_type", "cell_type")

    def train(self):
        logger.info("Generating random embedding")
        n_cells = self.dataset.n_obs
        self.dataset.obsm[self.latent_key] = np.random.randn(n_cells, self.n_dims)

    def _compute_model_metrics(self) -> dict:
        return {"n_dims": self.n_dims}


def main():
    setup_container_logging()
    job_spec = load_job_spec()
    config = build_model_config(model_name="random_embedding", job_spec=job_spec)
    seed = config.get("seed") or 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    mudata_obj = load_input_mudata()
    dataset_name = job_spec.get("dataset_name", "dataset")

    from ..data_utils import anndata_concatenate
    data = anndata_concatenate(
        list_anndata=[mudata_obj[m] for m in mudata_obj.mod.keys()],
        list_modality=list(mudata_obj.mod.keys()),
    )

    model = RandomEmbeddingModel(dataset=data, dataset_name=dataset_name, config_path=config)
    model.train()
    model.save_latent()
    model.umap()
    model.evaluate_model()

if __name__ == "__main__":
    main()
```

#### Step 3 — Write the environment file

Create `store/models/random_embedding/container/environment.yml`:

```yaml
name: multiverse_random_embedding
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.12
  - numpy
  - scanpy
  - mudata
  - h5py
  - matplotlib
  - pip
  - pip:
    - multiverse-worker  # thin worker interface (no orchestration code)
```

#### Step 4 — Write the Dockerfile

Create `store/models/random_embedding/container/Dockerfile`:

```dockerfile
FROM mambaorg/micromamba:2.3.0
USER root
WORKDIR /app

COPY store/models/random_embedding/container/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_random_embedding/bin:$PATH

# Install ONLY the worker interface (runtime_io, base, logging) — not the full platform
COPY multiverse/worker/ /app/multiverse/worker/
COPY multiverse/models/random_embedding.py /app/multiverse/models/
COPY multiverse/models/__init__.py /app/multiverse/models/
RUN pip install --no-deps -e .

ENTRYPOINT ["python", "-m", "multiverse.models.random_embedding"]
```

#### Step 5 — Write the model manifest

Create `store/models/random_embedding/model.yaml`:

```yaml
name: Random Embedding Baseline
slug: random_embedding
version: "1.0.0"
supported_omics:
  - rna
  - atac
  - adt
runtime:
  image: multiverse-random-embedding:1.0.0
build:
  context: container
  dockerfile: Dockerfile
```

#### Step 6 — Write the hyperparameter schema

Create `schemas/models/random_embedding.hyperparameters.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema",
  "title": "RandomEmbedding Hyperparameters",
  "type": "object",
  "properties": {
    "n_dims": {
      "type": "integer",
      "default": 32,
      "minimum": 2,
      "description": "Dimensionality of the random embedding"
    }
  }
}
```

#### Step 7 — Build, register, and run

```bash
# Build the image
docker build -f store/models/random_embedding/container/Dockerfile \
             -t multiverse-random-embedding:1.0.0 .

# Register with the orchestrator
make register-model manifest=store/models/random_embedding/model.yaml

# Add to run_manifest.yaml
cat >> run_manifest.yaml << 'EOF'
- dataset_slug: pbmc10k
  model_name: random_embedding
  model_params:
    n_dims: 32
EOF

# Run
make benchmark MANIFEST=run_manifest.yaml
```

You should see `random_embedding` appear in the run summary table with status SUCCESS.

---

### How-to Guide: Add an R Model

The container I/O contract is language-agnostic. The model only needs to read
`/output/job_spec.json` and write `/output/metrics.json` and `/output/embeddings.h5`.

**1. Write your R model script** (`store/models/my_r_model/container/model.R`):

```r
library(jsonlite)
library(rhdf5)

job_spec <- fromJSON("/output/job_spec.json")
seed <- job_spec$seed %||% 42L
set.seed(seed)

# Load data — use your preferred H5MU/H5AD reader
# ... train your model ...

# Write embeddings (n_cells × n_dims matrix)
h5createFile("/output/embeddings.h5")
h5write(latent_matrix, "/output/embeddings.h5", "latent")

# Write metrics
write_json(list(my_metric = 0.87), "/output/metrics.json", auto_unbox = TRUE)
```

**2. Use the shell entrypoint wrapper** to inject seed and config:

```dockerfile
COPY store/models/my_r_model/container/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
```

```bash
#!/bin/bash
# entrypoint.sh
set -euo pipefail
export MVR_SEED=$(python3 -c \
  "import json; print(json.load(open('/output/job_spec.json'))['seed'])")
Rscript /app/model.R
```

**3. Register and run** exactly as in the Python tutorial above.

---

### How-to Guide: Add Optuna Hyperparameter Sweep for Your Model

Add a `search_space` block to your job in `run_manifest.yaml`:

```yaml
- dataset_slug: pbmc10k
  model_name: random_embedding
  mode: sweep
  optimize_metric: silhouette_label
  direction: maximize
  n_trials: 20
  study_storage: sqlite:///store/optuna.db
  search_space:
    n_dims:
      type: int
      low: 8
      high: 128
      log: true
```

The orchestrator handles Optuna trial dispatch automatically. Each trial writes its metrics
to `metrics.json`, which Optuna reads to decide the next trial's hyperparameters.

---

### Explanation: Why Models Are Containers

**Isolation.** Each model has its own Conda environment with pinned package versions.
PCA uses `scikit-learn`; MultiVI uses `scvi-tools 1.x`; MOFA uses a specific `mofapy2` version.
These often conflict. Containers mean you can upgrade one model's dependencies without
touching any other.

**Language agnosticism.** The platform has already been extended with Python models.
The container contract (read `/input/data.h5mu`, write `/output/embeddings.h5`) is trivially
implementable in R, Julia, or any language with HDF5 bindings.

**Reproducibility.** The `image_digest` in the `models` registry table records the exact
image SHA256. Combined with `manifest_hash` in `runs`, you can reconstruct the exact
software environment for any historical run.

**What containers do NOT do.** Containers do not connect to the internet, touch the
SQLite registry, or know anything about other models running concurrently. They read a
flat JSON file and write flat output files. This is intentional — it keeps the contract
minimal and makes containers testable in complete isolation.

---

### Reference: Model Manifest Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Human-readable display name |
| `slug` | string | yes | Machine identifier, lowercase, no spaces (e.g. `pca`) |
| `version` | string | yes | Semantic version string (e.g. `1.0.0`) |
| `supported_omics` | list[string] | yes | Omics this model can handle. Any subset of `["rna","atac","adt","methylation"]`. Empty list = accepts any. |
| `runtime.image` | string | yes | Docker image tag (e.g. `multiverse-pca:1.0.0`) |
| `build.context` | string | no | Build context path relative to `model.yaml`. If absent, image is pulled remotely. |
| `build.dockerfile` | string | no | Dockerfile path relative to `build.context` |
| `contract_version` | string | no | Version of the container I/O contract. Defaults to `1.0.0`. |

### Reference: `job_spec.json` Fields Readable Inside a Container

| Field | Type | Description |
|---|---|---|
| `seed` | integer | Random seed to apply before any model initialization |
| `dataset_name` | string | Human-readable dataset name for logging |
| `model_name` | string | Model slug |
| `hyperparameters` | object | Flat or nested dict of model parameters from `run_manifest.yaml` |
| `metrics.model_metrics` | list[string] | Which model-specific metrics to compute. Absent = compute all defaults. |
| `metrics.bio_conservation` | list[string] | scib-metrics bio conservation metrics requested |
| `metrics.batch_correction` | list[string] | scib-metrics batch correction metrics requested |

### Reference: Required Output Files

| File | Required | Description |
|---|---|---|
| `/output/embeddings.h5` | **Yes** | HDF5 file with dataset `latent`, shape `(n_cells, n_dims)`, dtype float32 or float64 |
| `/output/metrics.json` | **Yes** | Flat JSON dict of scalar floats. Empty dict `{}` is valid. |
| `/output/umap.png` | No | UMAP visualization. Skipped silently if absent. |

---

## Development Workflow

### Running Tests

```bash
make test                          # full unit suite
uv run pytest tests/unit/ -v      # verbose, unit only
uv run pytest tests/integration/  # requires Docker
```

### Code Style

```bash
uv run ruff check multiverse/     # linting
uv run ruff format multiverse/    # formatting
```

### Adding a Unit Test for Your Model

Minimal test for seed behaviour:

```python
# tests/unit/test_seeds.py (add alongside existing tests)
from unittest.mock import MagicMock, patch

def test_random_embedding_sets_seed():
    with patch("random.seed") as mock_random, \
         patch("numpy.random.seed") as mock_np:
        with patch("multiverse.models.random_embedding.load_job_spec",
                   return_value={"seed": 99, "dataset_name": "test",
                                 "hyperparameters": {"n_dims": 8}}):
            with patch("multiverse.models.random_embedding.load_input_mudata",
                       return_value=MagicMock(mod={"rna": MagicMock()})):
                with patch("multiverse.models.random_embedding.anndata_concatenate",
                           return_value=MagicMock(n_obs=100, var=MagicMock(keys=lambda: []),
                                                  obsm={}, uns={})):
                    import multiverse.models.random_embedding as mod
                    mod.main()
        mock_random.assert_called_with(99)
        mock_np.assert_called_with(99)
```
