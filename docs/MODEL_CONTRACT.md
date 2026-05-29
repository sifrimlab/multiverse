# Model Contract

The authoritative runtime contract is [Model Container Contract](MODEL_CONTAINER_CONTRACT.md).

Use this page as the short conceptual summary: a model in mvexp is acceptable when it can read the standard input, honor the standard job specification, and write the standard outputs without knowing anything about the user's filesystem.

```mermaid
flowchart LR
    A[/input/data.h5mu] --> C[Model]
    B[/output/job_spec.json] --> C
    C --> D[/output/embeddings.h5]
    C --> E[/output/metrics.json]
```

## Zero-Path Promise

Researchers should never have to rewrite a model command because one dataset lives in a different directory from another. mvexp handles the path mapping; the model uses `/input/data.h5mu` and `/output/`.

## Minimum Hello World Model

```python
import json
import h5py
import numpy as np

with open("/output/job_spec.json") as f:
    spec = json.load(f)

rng = np.random.default_rng(spec.get("seed", 42))
latent = rng.normal(size=(100, 5)).astype("float32")

with h5py.File("/output/embeddings.h5", "w") as f:
    f.create_dataset("latent", data=latent)

with open("/output/metrics.json", "w") as f:
    json.dump({"hello_world_metric": 1.0}, f)
```

## Common Errors

See [Model Container Contract](MODEL_CONTAINER_CONTRACT.md#failure-modes).
