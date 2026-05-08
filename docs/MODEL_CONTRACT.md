# Model Container Contract

Version: 1.0

This document is the authoritative specification for any program that runs as a Multi-verse model container. It is intentionally **language-agnostic** — a compliant container can be implemented in Python, R, C++, Julia, or any other language.

---

## Principle

A model container is a black box. It knows nothing about the orchestrator, the registry, or other models. It receives data through filesystem mounts and writes results to a fixed output path. That is the entire contract.

---

## Mount Points

| Mount | Mode | Description |
|-------|------|-------------|
| `/input/data.h5mu` | read-only | MuData input file (h5mu format) containing pre-processed multimodal single-cell data |
| `/output/` | read-write | All container output must be written here |

The container **must not** access any path outside these two mount points.

---

## Input: `/output/job_spec.json`

Before the container starts, the orchestrator writes a JSON file to `/output/job_spec.json`. The container reads this to obtain its runtime configuration.

```json
{
  "seed": 42,
  "dataset_id": 1,
  "dataset_name": "pbmc10k",
  "model_name": "pca",
  "hyperparameters": {
    "n_components": 50,
    "device": "cpu"
  },
  "run_settings": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `seed` | integer | Random seed for reproducibility |
| `dataset_name` | string | Human-readable dataset identifier |
| `model_name` | string | Model identifier (matches the container's own name) |
| `hyperparameters` | object | Model-specific parameters (see per-model schemas in `schemas/models/`) |
| `run_settings` | object | Reserved for orchestrator-level overrides |

---

## Required Output: `/output/embeddings.h5`

The container **must** produce this file before exiting with code 0.

**Format:** HDF5

**Schema:**
```
/
└── latent   [dataset, float32 or float64, shape (n_cells, n_dims)]
```

- The dataset key is always `"latent"`.
- `n_cells` must match the number of observations in the input MuData.
- `n_dims` is the model's chosen latent dimensionality.
- The file should be written atomically (write to a `.tmp` path, then rename) to prevent the orchestrator from reading a partially written file.

**Example (Python with h5py):**
```python
import h5py, os, numpy as np

latent = np.random.randn(3000, 20).astype(np.float32)  # your embeddings
tmp = "/output/embeddings.h5.tmp"
with h5py.File(tmp, "w") as f:
    f.create_dataset("latent", data=latent)
os.rename(tmp, "/output/embeddings.h5")
```

**Example (R with hdf5r):**
```r
library(hdf5r)
f <- H5File$new("/output/embeddings.h5.tmp", mode = "w")
f[["latent"]] <- latent_matrix   # matrix with shape [n_cells, n_dims]
f$close_all()
file.rename("/output/embeddings.h5.tmp", "/output/embeddings.h5")
```

---

## Optional Outputs

The following files may be written to `/output/` but are not required:

| File | Description |
|------|-------------|
| `run.log` | Execution log (text) |
| `umap.png` | UMAP visualisation of the latent space |

The orchestrator ignores any file it does not recognise.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success — orchestrator will promote `/output/` to permanent storage |
| non-zero | Failure — orchestrator preserves the workspace directory for debugging |

---

## Rules

1. **No network access.** Containers run without internet connectivity at inference time.
2. **No knowledge of the orchestrator.** Do not import, call, or depend on the `multiverse` package or any other orchestrator-specific code.
3. **Write only to `/output/`.** Do not write to `/tmp`, `/var`, home directories, or any path outside `/output/`.
4. **`/input/data.h5mu` is read-only.** Do not modify or delete it.
5. **`/output/embeddings.h5` is mandatory.** A container that exits 0 without writing this file is considered failed by the orchestrator.
6. **Determinism via seed.** Read the `seed` field from `job_spec.json` and use it to seed all random number generators before training.

---

## Python SDK (Optional)

For Python-based models, the `mvr-worker` package provides convenience wrappers for all contract I/O:

```python
from mvr_worker import (
    setup_container_logging,
    load_job_spec,
    load_input_mudata,
    build_model_config,
    anndata_concatenate,
    save_embeddings,
)
```

Install inside the container:
```dockerfile
COPY sdk/mvr-worker/ /tmp/mvr-worker/
RUN pip install /tmp/mvr-worker/
```

In the future this will be published to PyPI as `pip install mvr-worker`.

**`save_embeddings(latent_array, output_dir="/output")`** handles atomic write and logging automatically.

The SDK has zero ML training dependencies (no torch, scanpy, scib-metrics). It only requires `h5py`, `numpy`, `mudata`, and `anndata`.

---

## Verifying Compliance

```bash
# After a run, verify the output file
python - <<'EOF'
import h5py, sys
with h5py.File("results/.../embeddings.h5", "r") as f:
    latent = f["latent"][:]
    print(f"Shape: {latent.shape}  dtype: {latent.dtype}")
    assert latent.ndim == 2, "Must be 2-D"
    assert latent.shape[0] > 0, "Must have at least one cell"
    assert latent.shape[1] > 0, "Must have at least one dimension"
print("Contract satisfied.")
EOF
```
