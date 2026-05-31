# Getting Started

This tutorial walks through a first mvexp benchmark, from a Jupyter-prepared object to a model embedding you can inspect again in Scanpy. The guiding idea is to keep the biology in your notebook and let mvexp handle the repeatable execution between curation and interpretation.

## What You Will Do

1. Save a small `AnnData` or `MuData` object from Jupyter.
2. Register it in the Streamlit GUI.
3. Configure a benchmark plan.
4. Launch the run.
5. Read `embeddings.h5` back into Jupyter.

## Before You Start

Install dependencies, initialize the registry, optionally start observability services, then launch the GUI:

```bash
make bootstrap      # uv sync --group dev + init registry + register built-in models
make register-all-datasets # add all the datasets
make services-up    # optional: MLflow on :5000, Optuna Dashboard on :8080
make setup          # optional: GUI and ML model wrapper extras (Streamlit, Scanpy, scvi-tools)
make gui            # Streamlit on :8501
```

Open `http://localhost:8501`. You do not need to run `docker` commands by hand during normal use; the mvd-backed runner manages model containers on your behalf.

The same setup can be driven directly through the canonical CLI:

```bash
uv run multiverse init-db
uv run multiverse register-model --slug pca
uv run multiverse register-model --slug multivi
uv run multiverse register-dataset --slug pbmc_rna
uv run multiverse run --manifest run_manifest.yaml --output store/artifacts/run_output
```

## Step 1: Prepare Data in Jupyter

For a single-modality RNA baseline:

```python
from pathlib import Path
import scanpy as sc

# adata = sc.read_h5ad("my_project/processed_pbmc.h5ad")

adata.obs["batch"] = adata.obs["donor_id"].astype(str)
adata.obs["cell_type"] = adata.obs["manual_annotation"].astype(str)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

dataset_dir = Path("store/datasets/pbmc_rna")
(dataset_dir / "data").mkdir(parents=True, exist_ok=True)
adata.write_h5ad(dataset_dir / "data" / "rna.h5ad")
```

For multimodal RNA+ATAC data, save a `MuData` object:

```python
from pathlib import Path
import mudata as md

dataset_dir = Path("store/datasets/pbmc_multiome")
(dataset_dir / "data").mkdir(parents=True, exist_ok=True)

mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})
mdata.obs["batch"] = adata_rna.obs["donor_id"].astype(str)
mdata.obs["cell_type"] = adata_rna.obs["cell_type"].astype(str)
mdata.write_h5mu(dataset_dir / "data" / "processed.h5mu")
```

## Step 2: Create a Dataset Manifest

The manifest describes what you saved and lets mvexp register the dataset consistently.

```python
import yaml

manifest = {
    "name": "PBMC RNA",
    "omics": ["rna"],
    "raw_files": {"rna": "data/rna.h5ad"},
    "metadata_keys": {"batch": "batch", "cell_type": "cell_type"},
}
with open("store/datasets/pbmc_rna/dataset.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

The `batch` key identifies the technical or donor grouping that batch-correction metrics will evaluate. The `cell_type` key identifies biological labels used by supervised metrics. If either column is absent, mvexp logs the value `"unknown"` for affected cells and disables the metrics that depend on the missing column. It does not silently invent biological labels.

See [Data Preparation](DATA_PREPARATION.md) for additional recipes (RNA+ATAC, RNA+ADT).

## Step 3: Register the Dataset

In the GUI:

1. Open the **Registry** tab.
2. Expand **Register New Dataset**.
3. Enter `store/datasets/pbmc_rna/dataset.yaml` in **Path to dataset.yaml**, or switch on **Build manifest from fields** and fill the form.
4. Click **Register Dataset**, then **Refresh Registry**.
5. Confirm your dataset appears with status `READY`.

The CLI equivalent, useful for scripted workflows:

```bash
make register slug=pbmc_rna
# or
uv run multiverse register-dataset --slug pbmc_rna
```

## Step 4: Configure the Benchmark

1. Open the **Configure** tab.
2. Review the compatibility matrix. Only `Compatible` cells are selectable.
3. Select the dataset × model pairs you want to run.
4. Adjust hyperparameters in the per-row forms — typed controls are rendered from each model's JSON schema.
5. Optionally toggle a parameter into a sweep distribution (requires `run_gridsearch: true` in globals).
6. Enter an experiment name and a random seed.
7. Click **Generate Run Manifest**.

The resulting `run_manifest.yaml` is part of your scientific record. See [Run Manifest](RUN_MANIFEST.md) for the schema.

## Step 5: Launch and Monitor

In the GUI:

1. Open the **Run** tab.
2. Confirm the manifest path and output directory, usually `store/artifacts/run_output`.
3. Click **Launch Run**.
4. Watch the status table. Jobs cycle through kernel states such as `PENDING -> RUNNING -> PROMOTING -> ARTIFACT_SUCCESS`, or `FAILED` / `CANCELLED`.

From the CLI:

```bash
uv run multiverse run --manifest run_manifest.yaml --output store/artifacts/run_output
```


## Step 6: Inspect Results

1. Open the **Results** tab.
2. Filter by experiment, dataset, model, or status.
3. Select a run to view metrics, the model log, `job_spec.json`, and the artifact tree.
4. Copy the artifact directory for notebook analysis.

The artifact layout is:

```text
<output-dir>/store/artifacts/<artifact-id>/
  artifact_manifest.json
  artifact_manifest.sha256
  job_spec.json
  embeddings.h5
  metrics.json        # optional
  umap.png            # optional
  run.log             # model SDK log (mvr_worker)
  container.log       # host-captured container stdout/stderr
  orchestrator.log    # host-side run reasoning (state transitions, failures)
```

### Where logs live

Each run carries up to three logs, surfaced together under **Logs** in the Results tab:

| File | Written by | Use it to debug |
|---|---|---|
| `run.log` | The model container via `mvr_worker` | Model-internal progress, metrics, warnings. |
| `container.log` | The host (captured container stdout/stderr) | Crashes, tracebacks, OOMs, or non-SDK images that never wrote `run.log`. |
| `orchestrator.log` | The host executor | Admission, launch, exit code, promotion outcome, and the exact failure reason. |

Successful runs are promoted to `store/artifacts/<artifact-id>/`. Runs that fail before promotion keep their logs in the run's workspace at `<output-dir>/store/workspaces/<attempt-id>/`, and cancelled runs under `<output-dir>/store/cancelled/<date>/<attempt-id>/`. Session-wide CLI events are written to `<output-dir>/multiverse.log`, and kernel state-machine events to `<output-dir>/journal/current.log`.

Set `MVEXP_LOG_LEVEL=DEBUG` (a level name or numeric value) before launching to raise verbosity across the host logs and the in-container `run.log`.

For cross-run comparison and metric histories, open the **Analysis** tab or visit MLflow at `http://localhost:5000` directly.

## Step 7: Bring Embeddings Back to Jupyter

```python
from pathlib import Path
import h5py
import scanpy as sc

artifact_dir = Path("store/artifacts/run_output/store/artifacts/<artifact-id>")
# Copy the exact path from the Results tab.

with h5py.File(artifact_dir / "embeddings.h5", "r") as f:
    embedding = f["latent"][:]

adata = sc.read_h5ad("store/datasets/pbmc_rna/data/rna.h5ad")
adata.obsm["X_mvexp_pca"] = embedding

sc.pp.neighbors(adata, use_rep="X_mvexp_pca")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

## Common Issues

| Symptom | Likely cause | What to do |
|---|---|---|
| Dataset does not appear in Configure | Registry has not refreshed. | Registry → **Refresh Registry**. |
| Job is `FAILED` | Docker launch, container execution, or output validation failed. | Open `orchestrator.log` for the failure reason, then `container.log` for the container traceback. For failed runs these stay under `store/workspaces/<attempt-id>/`. |
| `executor crashed: unverified_local` | Running with `--strict` but image has no registry digest. | Remove `--strict`. The default run allows locally-built images. |
| Metric is missing | `batch_key` or `cell_type_key` does not support that metric. | Confirm columns exist in your `obs`; re-register if you fix them. |
| `database is locked` | Concurrent registry writes or an interrupted process. | Retry. If Results looks stale, run `uv run multiverse rebuild-index --state-root store/artifacts/run_output --store-root store/artifacts/run_output/store`. |

## Writing Your Methods Section

For a publication, keep these artifacts with the analysis:

- `run_manifest.yaml`: datasets, models, parameters, seed, metric selection.
- `job_spec.json`: exact per-job runtime instruction passed to the model container.
- `metrics.json`: model metrics and training histories where available.
- `run.log` / `container.log`: model and host-captured execution logs.
- `provenance.json`: additional provenance when present.

A Methods paragraph can state:

> Integration benchmarks were run with mvexp (commit `<sha>`). Datasets were registered with batch key `batch` and cell-type key `cell_type`. The benchmark plan, model parameters, random seed, and metric configuration are provided in Supplementary File X (`run_manifest.yaml`). Per-model runtime specifications and output provenance are archived with each run artifact.

## Where to Go Next

- [Data Preparation](DATA_PREPARATION.md) — recipes for RNA, RNA+ATAC, RNA+ADT.
- [Models Glossary](reference/MODELS_GLOSSARY.md) — assumptions and hyperparameters per model.
- [Evaluation Metrics](reference/EVALUATION_METRICS.md) — what each metric measures.
- [Benchmarking](BENCHMARKING.md) — designing a defensible comparison.
