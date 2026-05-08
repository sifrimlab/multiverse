# Model Registration Guide (Contract Over Code)

> We don't care if your model is written in Python, R, or Julia.  
> As long as your Docker container reads `/input/data.h5mu` and writes its embeddings to `/output/embeddings.h5`, it is a first-class citizen in the Multiverse.

This guide defines the single source of truth between orchestrator and model containers.

## Required Model Package (Before Registration)

`make register-model` only succeeds in practice when the full model package is present, not just `model.yaml`.

For a model slug `<slug>`, provide:

- `store/models/<slug>/model.yaml` (manifest metadata)
- `store/models/<slug>/container/Dockerfile` (container build recipe)
- `store/models/<slug>/container/environment.yml` (runtime dependencies)
- `schemas/models/<slug>.hyperparameters.schema.json` (GUI + tuning parameter contract)
- model runtime entry script/module (for example `multiverse/models/<slug>.py`, referenced by your Docker image `CMD`/`ENTRYPOINT`)

If any of these are missing, registration may complete but execution/build/GUI parameter rendering will break.

## Zero-Path Container Contract

The orchestrator mounts and manages exactly these paths:

- **Input (read-only):** `/input/data.h5mu`
- **Output (read-write):** `/output/`
- **Job spec file:** `/output/job_spec.json`

Container responsibilities:

1. Read runtime parameters from `/output/job_spec.json`.
2. Read data from `/input/data.h5mu`.
3. Write outputs to `/output/` (never assume host paths).

Required output artifacts:

- `/output/embeddings.h5`
- `/output/metrics.json`

Common optional outputs:

- `/output/umap.png`
- `/output/container.log` (usually written by orchestrator log capture)

## `model.yaml` Manifest

Each model lives under:

`store/models/<slug>/`

Example:

```yaml
name: PCA
version: 1.0.0
contract_version: 1.0.0
supported_omics: ["rna"]
runtime:
  image: multiverse-pca:1.0.0
hyperparameters_schema: schemas/models/pca.hyperparameters.schema.json
build:
  context: ../../..
  dockerfile: store/models/pca/container/Dockerfile
```

### Key Fields

- `name`: display/model name.
- `version`: semantic version for registry keying.
- `supported_omics`: modalities required by the model (`["any"]` allowed by schema rules).
- `runtime.image`: Docker image reference with explicit tag.
- `hyperparameters_schema`: JSON schema path used by planning/tuning tools.
- `hyperparameters_schema` also drives Streamlit GUI field generation (typed widgets, defaults, and enums).
- `build` (optional): local Docker build recipe:
  - `context`
  - `dockerfile`

If `build` is omitted, the orchestrator assumes a prebuilt/remote image reference.

## Step-by-Step: Add a New Model

1. **Create folder**
   - `store/models/<slug>/`
2. **Add container assets**
   - `store/models/<slug>/container/Dockerfile`
   - `store/models/<slug>/container/environment.yml`
3. **Add model runtime script**
   - Implement the model entrypoint script/module used by the container.
   - Ensure it reads `/input/data.h5mu` and `/output/job_spec.json`, then writes artifacts to `/output/`.
4. **Add hyperparameter schema**
   - `schemas/models/<slug>.hyperparameters.schema.json`
   - Keep property names aligned with keys expected by the model script.
5. **Write `model.yaml`**
   - Include `runtime.image`, `build.dockerfile`, and `hyperparameters_schema`.
6. **Register**
   - `make register-model slug=<slug>`
   - Or bulk-register built-ins: `make register-models`
7. **(Optional) build locally**
   - `mvr models build --slug <slug>`

## Local Build Engine

Multiverse includes a local builder for private iteration:

- resolves `build.context` and `build.dockerfile` relative to `model.yaml`
- builds using Docker SDK with native cache behavior
- tags image exactly as `runtime.image`
- does **not** push to any registry

Commands:

```bash
make register-model slug=pca
mvr models build --slug pca
```

