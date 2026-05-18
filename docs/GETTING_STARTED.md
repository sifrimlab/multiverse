# Getting Started

This tutorial walks through a first mvexp benchmark from a Jupyter-prepared object to a model embedding you can inspect again in Scanpy.

The guiding idea is: keep the biology in your notebook, and let mvexp handle the repeatable execution.

## What You Will Do

You will:

1. Save a small `AnnData` or `MuData` object from Jupyter.
2. Register it in the Streamlit GUI.
3. Build a benchmark plan.
4. Launch the run.
5. Read `embeddings.h5` back into Jupyter.

## Before You Start

Install and start mvexp:

```bash
make bootstrap
make services-up
make setup
```

Open `http://localhost:8501`.

You do not need to run Docker commands by hand during normal use. The GUI and orchestrator take care of that boundary for you.

## Step 1: Prepare Data in Jupyter

For a single-modality RNA baseline such as PCA, save an `AnnData` object.

```python
from pathlib import Path

import scanpy as sc

# Start from an AnnData object you already curated.
# adata = sc.read_h5ad("my_project/processed_pbmc.h5ad")

adata.obs["batch"] = adata.obs["donor_id"].astype(str)
adata.obs["cell_type"] = adata.obs["manual_annotation"].astype(str)

# PCA-style baselines are commonly run on normalized/log-transformed data.
# Keep your raw counts in adata.raw or a layer if your study needs them.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

dataset_dir = Path("store/datasets/pbmc_rna")
data_dir = dataset_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

adata.write_h5ad(data_dir / "rna.h5ad")
```

For multimodal RNA+ATAC data, save a `MuData` object.

```python
from pathlib import Path

import mudata as md

# adata_rna and adata_atac should contain the same cell barcodes
# or be aligned according to your study design.

dataset_dir = Path("store/datasets/pbmc_multiome")
data_dir = dataset_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

mdata = md.MuData({
    "rna": adata_rna,
    "atac": adata_atac,
})

mdata.obs["batch"] = adata_rna.obs["donor_id"].astype(str)
mdata.obs["cell_type"] = adata_rna.obs["cell_type"].astype(str)

mdata.write_h5mu(data_dir / "processed.h5mu")
```

## Step 2: Create a Dataset Manifest

The manifest is a short description of what you saved. It lets mvexp register the dataset consistently.

For RNA-only data:

```python
import yaml

manifest = {
    "name": "PBMC RNA",
    "omics": ["rna"],
    "raw_files": {
        "rna": "data/rna.h5ad",
    },
    "metadata_keys": {
        "batch": "batch",
        "cell_type": "cell_type",
    },
}

with open("store/datasets/pbmc_rna/dataset.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

For RNA+ATAC data saved as a single `processed.h5mu`:

```python
import yaml

manifest = {
    "name": "PBMC Multiome",
    "omics": ["rna", "atac"],
    "raw_files": {
        "rna": "data/processed.h5mu",
        "atac": "data/processed.h5mu",
    },
    "metadata_keys": {
        "batch": "batch",
        "cell_type": "cell_type",
    },
}

with open("store/datasets/pbmc_multiome/dataset.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

The `batch` key should identify the technical or donor grouping you want batch-correction metrics to evaluate. The `cell_type` key should identify biological labels for supervised metrics.

## Step 3: Register the Dataset in the GUI

1. Open the **Registry** tab.
2. Expand **Register New Dataset**.
3. Leave **Build manifest from fields** switched off if you already wrote `dataset.yaml`.
4. In **Path to dataset.yaml**, enter `store/datasets/pbmc_rna/dataset.yaml` or your own manifest path.
5. Click **Register Dataset**.
6. Click **Refresh Registry**.
7. Confirm your dataset appears in the Datasets table with status `READY`.

If you prefer not to write the manifest in Jupyter:

1. Open the **Registry** tab.
2. Expand **Register New Dataset**.
3. Switch on **Build manifest from fields**.
4. Enter dataset name, available omics, file paths, `batch_key`, and `cell_type_key`.
5. Click **Register Dataset**.
6. Click **Refresh Registry**.

## Step 4: Build a Benchmark Plan

1. Open the **Job Builder** tab.
2. Review the compatibility matrix.
3. Select the dataset x model pairs you want to run.
4. Enter an experiment name.
5. Click **Generate Run Manifest**.
6. Review the generated `run_manifest.yaml` shown in the page.

This manifest is part of your scientific record. It captures the models, datasets, parameters, metrics, and seed for the run.

## Step 5: Set Parameters

1. Open the **Parameters** tab.
2. Expand each selected dataset x model pair.
3. Adjust model parameters such as `n_components`, `n_factors`, `latent_dimensions`, `learning_rate`, or training epochs.
4. If you want a sweep, enable the sweep controls for the relevant parameter.
5. Click **Generate Run Manifest (with params)**.

The GUI uses the registered model schema to show typed controls. You do not need to memorize YAML syntax for model parameters.

## Step 6: Launch and Monitor

1. Open the **Execute** tab.
2. Confirm the manifest path points to `run_manifest.yaml`.
3. Choose an output directory if needed.
4. Set a random seed.
5. Click **Launch Run**.
6. Watch the status table for running, successful, failed, or skipped jobs.
7. Use the live metrics panel if MLflow services are running.

Skipped jobs are usually informative rather than mysterious: the most common reasons are incompatible omics, a missing `batch` column, or metadata that cannot support a requested metric.

## Step 7: Inspect Results

1. Open the **Results** tab.
2. Filter to `SUCCESS` runs.
3. Select a run.
4. Review the metrics table, model log, `job_spec.json`, and artifact directory.
5. Copy the artifact directory path for notebook analysis.

The usual artifact layout is:

```text
store/artifacts/<experiment>/<dataset>/<model>/<run_id>/
  run_manifest.yaml
  job_spec.json
  embeddings.h5
  metrics.json
  umap.png
  container.log
```

## Step 8: Bring Embeddings Back to Jupyter

```python
from pathlib import Path

import h5py
import scanpy as sc

artifact_dir = Path("store/artifacts/pbmc-benchmark/pbmc_rna/pca/run_abc123def456")

with h5py.File(artifact_dir / "embeddings.h5", "r") as f:
    embedding = f["latent"][:]

adata = sc.read_h5ad("store/datasets/pbmc_rna/data/rna.h5ad")
adata.obsm["X_mvexp_pca"] = embedding

sc.pp.neighbors(adata, use_rep="X_mvexp_pca")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

You can now continue with the plotting and biological interpretation tools you already use.

## Common Errors

| Symptom | Likely cause | What to do |
|---|---|---|
| Dataset does not appear in Job Builder | Registry has not refreshed. | Return to Registry and click **Refresh Registry**. |
| Job is `SKIPPED` | Incompatible omics or missing metadata. | Check the compatibility matrix and metadata keys. |
| Job is `FAILED` | Data cannot be read or model input requirements are not met. | Open Results and inspect the log for that run. |
| Metrics are missing | `batch_key` or `cell_type_key` does not support that metric. | Confirm metadata columns in Jupyter and re-register if needed. |

## Writing Your Methods Section

For a publication, keep these artifacts with the analysis:

- `run_manifest.yaml`: datasets, models, parameter values, metrics, seed, and experiment name.
- `job_spec.json`: exact per-job runtime instruction passed to a model.
- `metrics.json`: model metrics and histories where available.
- `container.log`: execution log for auditability.
- provenance artifacts in the run directory, when present.

A Methods paragraph can state:

```text
Integration benchmarks were run with mvexp. Datasets were registered with batch key
`batch` and cell-type key `cell_type`. The benchmark plan, model parameters, random seed,
and metric configuration are provided in Supplementary File X (`run_manifest.yaml`).
Per-model runtime specifications and output provenance are provided with each run artifact.
```

This is usually more precise than prose alone, because the manifest and provenance record the exact benchmark plan used to generate the figures.

## Cookbook Recipes to Add Next

- **RNA+ATAC PBMC Multiome Benchmark**: compare PCA, MOFA, MultiVI, Mowgli, and Cobolt on paired RNA+ATAC cells and interpret bio-conservation versus batch-correction tradeoffs.
- **CITE-seq RNA+ADT with TotalVI**: prepare protein expression metadata, run TotalVI, and visualize whether protein-informed embeddings improve annotated immune populations.
- **Atlas Subset Across Donors**: register donor as `batch`, compare integration models across multiple tissues or cohorts, and export a publication-ready provenance bundle.
