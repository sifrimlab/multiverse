# Model Container Contract

This reference defines the Zero-Path contract for any model container used by mvexp.

## Contract Summary

| Requirement | Value |
|---|---|
| Input data | `/input/data.h5mu` |
| Runtime spec | `/output/job_spec.json` |
| Output directory | `/output/` |
| Required embedding | `/output/embeddings.h5` with dataset `latent` |
| Required metrics | `/output/metrics.json` |

## `job_spec.json` Fields

| Field | Type | Meaning |
|---|---|---|
| `seed` | integer | Random seed for the run. |
| `dataset_id` | integer or null | Registry identifier when available. |
| `dataset_name` | string | Dataset display name or slug. |
| `model_name` | string | Model slug/name. |
| `hyperparameters` | object | Parameters selected in the GUI or manifest. |
| `run_settings` | object | Experiment-level settings. |

Example:

```json
{
  "seed": 42,
  "dataset_id": 1,
  "dataset_name": "hello_pbmc",
  "model_name": "pca",
  "hyperparameters": {"n_components": 20, "device": "cpu"},
  "run_settings": {"experiment_name": "hello-world"}
}
```

## Required `embeddings.h5` Format

```text
/
└── latent  shape: (n_cells, n_dimensions), dtype: float32 or float64
```

The number of rows must match the cells used by the model.

## Hello World Compliance Check

```python
import h5py

with h5py.File("/output/embeddings.h5", "r") as f:
    latent = f["latent"][:]

assert latent.ndim == 2
assert latent.shape[0] > 0
assert latent.shape[1] > 0
```

## Rules

1. Read data only from `/input/data.h5mu`.
2. Read parameters only from `/output/job_spec.json`.
3. Write model outputs only under `/output/`.
4. Apply the provided seed before training.
5. Exit non-zero when required inputs are invalid.
6. Do not require the researcher to provide host-specific paths.

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `embeddings.h5` missing | Model did not write required output. | Write `/output/embeddings.h5` before successful exit. |
| `latent` key missing | HDF5 file has wrong internal dataset name. | Use dataset name `latent`. |
| Metrics missing | `metrics.json` was not created. | Write JSON scalar diagnostics, even if minimal. |
| Host path dependency | Code expects local project paths. | Use Zero-Path inputs only. |

## Citation Note

For publication, archive the container version or image tag, `model.yaml`, `job_spec.json`, `run_manifest.yaml`, and provenance files with the run artifacts.
