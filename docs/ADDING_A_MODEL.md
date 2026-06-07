# Adding a Model

This how-to is for developers adding a new integration model to Multiverse. Once your model is registered, users should be able to select it in the GUI, set parameters, and receive standard artifacts.

## Goal

A good model behaves like this:

1. It appears in the Registry model table.
2. It is reported `Compatible` only against datasets that supply its required omics.
3. Its hyperparameters appear as typed controls on the **Configure** tab, derived from its JSON schema.
4. It runs under the container contract documented in [Model Container Contract](MODEL_CONTAINER_CONTRACT.md).
5. It writes the required artifacts (`embeddings.h5`, `metrics.json`, `umap.png`, `run.log`).
6. Its results are comparable to other models in the Results tab and in MLflow.

## Reference: Required Files

| File | Purpose |
|---|---|
| `store/models/<slug>/model.yaml` | Model metadata and registry contract. |
| `store/models/<slug>/hyperparameters.schema.json` | GUI controls and sweep definitions. |
| `store/models/<slug>/container/Dockerfile` | Runtime environment definition. |
| Runtime entrypoint | Reads Zero-Path inputs and writes required outputs. |

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
## Reference: `model.yaml` Fields

| Field | Required | Meaning |
|---|---|---|
| `name` | Yes | Display name in the GUI. |
| `version` | Yes | Model package version. |
| `contract_version` | Yes | Runtime contract version. |
| `supported_omics` | Yes | Modalities required by the model. Use `["any"]` for a modality-agnostic model compatible with every dataset (do not mix `any` with concrete modalities). |
| `runtime.image` | Yes | Image used by multiverse execution. |
| `hyperparameters_schema` | Recommended | JSON schema for GUI parameter controls. |
| `build.context` | Optional | Build context for maintainers. |
| `build.dockerfile` | Optional | Dockerfile path for maintainers. |

Minimal `model.yaml`:

```yaml
name: HelloModel
version: 1.0.0
contract_version: 1.0.0
supported_omics: ["rna"]
runtime:
  image: multiverse-hello-model:1.0.0
hyperparameters_schema: store/models/hello_model/hyperparameters.schema.json
build:
  context: ../../..
  dockerfile: store/models/hello_model/container/Dockerfile
```

## Live Per-Epoch Metrics (Optional but Recommended)

If your model trains iteratively, stream per-epoch metrics so they appear live in MLflow and survive crashes via a local `metrics.jsonl` sidecar. Use the thin helper `EpochLogger` exported from the `multiverse.worker` SDK — every model container already installs it. It logs to MLflow when `MLFLOW_TRACKING_URI` is set in the container environment (the runner propagates this automatically) and otherwise silently writes JSONL only.

Manual loop (PyTorch-style):

```python
from multiverse.worker import EpochLogger

with EpochLogger(
    jsonl_path="/output/metrics.jsonl",
    run_name=f"{dataset_name}-{model_name}",
) as ep:
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_loss = evaluate(...)
        ep.log(step=epoch, train_loss=train_loss, val_loss=val_loss)
```

`EpochLogger` does **not** replace writing `metrics.json` — your container should still write final scalars + a `history` block to `/output/metrics.json` as usual. The two are complementary: `metrics.json` is the final summary, `metrics.jsonl` is the live stream. See `store/models/cobolt/container/run.py` for a reference wiring.

## Citation Note

If the model corresponds to a published method, cite that method and Multiverse. Archive `model.yaml`, the hyperparameter schema, image tag/version, `run_manifest.yaml`, and provenance artifacts.
