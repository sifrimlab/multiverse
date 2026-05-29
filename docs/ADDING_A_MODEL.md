# Adding a Model

This how-to is for developers adding a new integration model to mvexp. It keeps the researcher experience visual: once your model is registered, users should be able to select it in the GUI, set parameters from typed controls, and receive standard artifacts.

## Goal

A good mvexp model behaves like this:

1. It appears in the Registry model table.
2. It is reported `Compatible` only against datasets that supply its required omics.
3. Its hyperparameters appear as typed controls on the **Configure** tab, derived from its JSON schema.
4. It runs under the container contract documented in [Model Container Contract](MODEL_CONTAINER_CONTRACT.md).
5. It writes the required artifacts (`embeddings.h5`, `metrics.json`, `umap.png`, `run.log`).
6. Its results are comparable to other models in the Results tab and in MLflow.

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

## Live Per-Epoch Metrics (Optional but Recommended)

If your model trains iteratively, stream per-epoch metrics so they appear live in MLflow and survive crashes via a local `metrics.jsonl` sidecar. Use the thin helper `EpochLogger` exported from the `mvr_worker` SDK — every model container already installs it. It logs to MLflow when `MLFLOW_TRACKING_URI` is set in the container environment (the runner propagates this automatically) and otherwise silently writes JSONL only.

Manual loop (PyTorch-style):

```python
from mvr_worker import EpochLogger

with EpochLogger(
    jsonl_path="/output/metrics.jsonl",
    run_name=f"{dataset_name}-{model_name}",
) as ep:
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_loss = evaluate(...)
        ep.log(step=epoch, train_loss=train_loss, val_loss=val_loss)
```

Framework with built-in history (scvi-tools, Cobolt, etc.) — replay after `.train()`:

```python
with EpochLogger(jsonl_path="/output/metrics.jsonl", run_name=run_name) as ep:
    length = max(len(v) for v in history.values())
    for step in range(length):
        ep.log(step=step, **{k: v[step] for k, v in history.items() if step < len(v)})
```

Keras: instantiate `EpochLogger` and pass an `on_epoch_end` callback that calls `ep.log(step=epoch, **logs)`. A copy-pasteable template is included at the bottom of `sdk/mvr-worker/mvr_worker/epoch_logger.py`.

`EpochLogger` does **not** replace writing `metrics.json` — your container should still write final scalars + a `history` block to `/output/metrics.json` as usual. The two are complementary: `metrics.json` is the final summary, `metrics.jsonl` is the live stream. See `store/models/cobolt/container/run.py` for a reference wiring.

**Single MLflow run per execution.** The host runner opens an MLflow run with hyperparameters + system-metrics monitoring *before* launching your container, and injects `MLFLOW_RUN_ID` into the container environment. `EpochLogger` detects that variable and attaches to the same run instead of starting a fresh one. After the container exits, the host appends final scalars + artifacts to that run and closes it with `FINISHED` or `FAILED`. You don't need to do anything special in your container — just call `EpochLogger(...)` as shown above. If `MLFLOW_RUN_ID` is absent (e.g. running your container manually outside the runner), `EpochLogger` falls back to creating its own run.

> **Rebuild required.** Because `mvr_worker` is `COPY`'d into the image at build time, any change to your model container or to the SDK takes effect only after rebuilding the image (`docker compose build <model>` or your usual image build).

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

## Building for HPC (Apptainer/Singularity)

HPC clusters typically ban Docker but support [Apptainer](https://apptainer.org/) (formerly Singularity). Multiverse supports both runtimes; this section covers the HPC authoring path.

### Singularity.def structure

Create `store/models/<slug>/container/Singularity.def` alongside your `Dockerfile`:

```singularity
Bootstrap: docker
From: mambaorg/micromamba:1.5.1

%files
    store/models/<slug>/container/environment.yml /opt/environment.yml
    sdk/ /opt/sdk/

%post
    micromamba install -y -n base -f /opt/environment.yml
    micromamba clean --all --yes
    pip install /opt/sdk/mvr-worker/ --no-deps

%runscript
    exec python /opt/sdk/mvr-worker/run.py "$@"
```

The `%runscript` **must** invoke the same entry point as the `Dockerfile`'s `CMD` so the container contract (`/input/data.h5mu`, `/output/job_spec.json`, `/output/`) is identical across backends.

Register the def file path in `model.yaml`:

```yaml
apptainer:
  build_from: def_file
  def_file: store/models/<slug>/container/Singularity.def
  gpu_required: false  # set to true for GPU models
```

### Building the SIF

**On a workstation with Docker + Apptainer:**

```bash
make build-sif slug=<slug>
# or with docker-daemon method explicitly:
multiverse build-sif --slug <slug> --method docker-daemon
```

**On an HPC login node (no Docker):**

```bash
multiverse build-sif --slug <slug> --method def-file
```

Transfer the resulting `.sif` file to the cluster if built locally.

### Testing the SIF locally

```bash
apptainer exec \
  --bind <data_path>:/input:ro \
  --bind <workspace>:/output \
  <slug>.sif python /opt/sdk/mvr-worker/run.py
```

### Registering the SIF path

After building, register the SIF path so `multiverse run --backend slurm` can resolve it automatically:

```bash
multiverse register-model --manifest store/models/<slug>/model.yaml --set-sif-path /path/to/<slug>-<version>.sif
```

### Running on Slurm

Pre-check your HPC environment:

```bash
multiverse doctor --deep-slurm
```

Then run via manifest:

```bash
multiverse run --manifest run_manifest.yaml --backend slurm --output /path/to/output
```

> **Note:** SIFs built by `multiverse build-sif` — whether from `--method
> def-file` or `--method docker-daemon` against a locally-built image — have no
> trustworthy registry provenance and are tagged `unverified_local`. Slurm runs
> with these SIFs require `--accept-degraded` (or `accept_degraded: true` in the
> kernel config). Strict acceptability comes only from a real registry digest
> source (an explicitly supplied/persisted `image_digest`), not from the fact
> that Apptainer built a SIF from Docker's local daemon. Persisting a registry
> digest for registry-sourced images is a separate, not-yet-implemented feature.

### Provenance note

| Build method | Image identity | accept-degraded required? |
|---|---|---|
| `docker-daemon` (local image) | `unverified_local` | **Yes** |
| `def-file` | `unverified_local` | **Yes** |
| supplied registry digest (`image_digest` in manifest) | `registry_digest` (strict) | No |
