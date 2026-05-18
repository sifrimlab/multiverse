# Model Containers

This explanation is for researchers and model authors who need to understand Zero-Path execution. Routine benchmarking does not require manual Docker commands.

## The Main Idea

Every model sees the same simple filesystem, regardless of where your project lives on the host machine.

```mermaid
flowchart LR
    A[Registered dataset] --> B[mvexp runner]
    B --> C[/input/data.h5mu]
    B --> D[/output/job_spec.json]
    C --> E[Model container]
    D --> E
    E --> F[/output/embeddings.h5]
    E --> G[/output/metrics.json]
```

[IMAGE: Zero-Path Container Contract]

## Why Researchers Should Care

Zero-Path execution reduces the common failure mode where one model expects files in a different place from another model. You select models in the GUI; mvexp gives each model the same input and output paths.

## Reference: Container Paths

| Path | Direction | Meaning |
|---|---|---|
| `/input/data.h5mu` | Read-only input | Prepared MuData object for the selected dataset. |
| `/output/job_spec.json` | Input instruction | Seed, dataset name, model name, parameters, and run settings. |
| `/output/embeddings.h5` | Required output | Latent matrix used for comparison and downstream analysis. |
| `/output/metrics.json` | Required output | Model diagnostics and metrics. |
| `/output/umap.png` | Optional output | Quick visualization. |

## Hello World Model Output

A compliant model must write `embeddings.h5` with a dataset named `latent`:

```python
import h5py
import numpy as np

latent = np.random.normal(size=(100, 10)).astype("float32")

with h5py.File("/output/embeddings.h5", "w") as f:
    f.create_dataset("latent", data=latent)
```

It must also write a metric file:

```python
import json

with open("/output/metrics.json", "w") as f:
    json.dump({"example_score": 1.0}, f, indent=2)
```

## Common Errors

| Symptom | Likely cause | What to do |
|---|---|---|
| Container succeeds but run is failed | Required `embeddings.h5` or `metrics.json` is missing. | Check the model writes both files under `/output/`. |
| Latent matrix shape error | Number of rows does not match cells. | Confirm row order and cell count in the input data. |
| File path error inside model | Model tried to read host paths. | Use only `/input/data.h5mu` and `/output/`. |
| Reproducibility differs across runs | Seed was not applied to all random generators. | Read `seed` from `job_spec.json` and apply it consistently. |

## Citation Note

Containerized execution is part of the reproducibility record. For papers, cite mvexp and archive the model version, `model.yaml`, and run artifacts used to generate the result.
