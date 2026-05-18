# Adding a Model

This how-to is for developers adding a new integration model to mvexp. It keeps the researcher experience visual: once your model is registered, users should be able to select it in the GUI, set parameters from typed controls, and receive standard artifacts.

## Goal

A good mvexp model behaves like this:

1. It appears in the Registry model table.
2. It appears as compatible only for supported omics.
3. Its hyperparameters appear in the Parameters tab.
4. It runs through Zero-Path execution.
5. It writes `embeddings.h5` and `metrics.json`.
6. Its outputs appear in comparison reports.

[IMAGE: Custom Model in Job Builder]

## Tutorial: Hello World Model

Minimal runtime script:

```python
import json
import h5py
import mudata as md
import numpy as np

with open("/output/job_spec.json") as f:
    spec = json.load(f)

mdata = md.read_h5mu("/input/data.h5mu")
n_cells = mdata.n_obs
n_components = spec.get("hyperparameters", {}).get("n_components", 5)
rng = np.random.default_rng(spec.get("seed", 42))
latent = rng.normal(size=(n_cells, n_components)).astype("float32")

with h5py.File("/output/embeddings.h5", "w") as f:
    f.create_dataset("latent", data=latent)

with open("/output/metrics.json", "w") as f:
    json.dump({"hello_world_score": 1.0}, f, indent=2)
```

Minimal `model.yaml`:

```yaml
name: HelloModel
version: 1.0.0
contract_version: 1.0.0
supported_omics: ["rna"]
runtime:
  image: mvexp-hello-model:1.0.0
hyperparameters_schema: schemas/models/hello_model.hyperparameters.schema.json
build:
  context: ../../..
  dockerfile: store/models/hello_model/container/Dockerfile
```

Minimal schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "HelloModel Hyperparameters",
  "type": "object",
  "properties": {
    "n_components": {"type": "integer", "minimum": 2, "default": 5},
    "device": {"type": "string", "enum": ["cpu", "cuda", "cuda:0"], "default": "cpu"}
  },
  "additionalProperties": false
}
```

## Reference: Required Files

| File | Purpose |
|---|---|
| `store/models/<slug>/model.yaml` | Model metadata and registry contract. |
| `schemas/models/<slug>.hyperparameters.schema.json` | GUI controls and sweep definitions. |
| `store/models/<slug>/container/Dockerfile` | Runtime environment definition. |
| Runtime entrypoint | Reads Zero-Path inputs and writes required outputs. |

## Reference: `model.yaml` Fields

| Field | Required | Meaning |
|---|---|---|
| `name` | Yes | Display name in the GUI. |
| `version` | Yes | Model package version. |
| `contract_version` | Yes | Runtime contract version. |
| `supported_omics` | Yes | Modalities required by the model. |
| `runtime.image` | Yes | Image used by mvexp execution. |
| `hyperparameters_schema` | Recommended | JSON schema for GUI parameter controls. |
| `build.context` | Optional | Build context for maintainers. |
| `build.dockerfile` | Optional | Dockerfile path for maintainers. |

## Explanation: Designing for Notebook-First Researchers

Researchers should not need to know how your model is launched. Put all user-facing choices in the hyperparameter schema, use clear parameter names, and write interpretable metrics. If a parameter would be hard to explain in a Methods section, document it in the schema description or model glossary.

## Common Errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Model does not show parameters | Missing or invalid schema. | Validate the JSON schema and `hyperparameters_schema` path. |
| Run cannot find input files | Model uses host paths. | Read only `/input/data.h5mu`. |
| Run is successful but no comparison | Missing `embeddings.h5` or malformed latent dataset. | Write `/output/embeddings.h5` with key `latent`. |
| Repeated runs differ unexpectedly | Seed not applied. | Use `seed` from `job_spec.json` for all random generators. |
| Metrics are hard to interpret | Only training loss is reported. | Add biologically meaningful diagnostics when available. |

## Citation Note

If the model corresponds to a published method, cite that method and mvexp. Archive `model.yaml`, the hyperparameter schema, image tag/version, `run_manifest.yaml`, and provenance artifacts.
