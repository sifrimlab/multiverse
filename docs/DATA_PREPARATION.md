# Data Preparation

This how-to explains how to prepare biological data for mvexp from the perspective of a Scanpy, Seurat, or MuData user.

The short version: mvexp expects a well-formed `AnnData` or `MuData` file, plus explicit metadata keys that say which column represents batch and which column represents biological labels.

## Data State: What mvexp Assumes

mvexp does not perform scientific preprocessing decisions for you. It preserves and records the choices you make before registration.

| Requirement | Practical guidance |
|---|---|
| Cells are observations | Cell barcodes should be in `.obs_names`. |
| Features are variables | Genes, peaks, or proteins should be in `.var_names`. |
| Modalities are explicit | Use `MuData` modalities such as `rna`, `atac`, and `adt` for multimodal studies. |
| `batch` metadata is present | Use a string or categorical `.obs` column for donor, chemistry, lane, site, or another nuisance source. |
| `cell_type` metadata is present when available | Use a string or categorical `.obs` column for biological labels used by ARI, NMI, silhouette, and related metrics. |
| Preprocessing is intentional | Store the matrix state expected by the selected models and document normalization, filtering, HVG selection, and feature construction. |

## Raw Counts, Normalized Data, and HVGs

Different models make different assumptions. mvexp records what you run, but it does not convert one scientific data state into another without your knowledge.

Recommended practice:

- Keep raw counts in a layer or adjacent file when possible.
- For PCA baselines, normalized and log-transformed matrices with HVG selection are often appropriate.
- For probabilistic count models, preserve count-like input unless the model documentation says otherwise.
- For ATAC, document whether the matrix is binary accessibility, counts, TF-IDF transformed, or peak-selected.
- For ADT, document whether values are raw counts, denoised counts, CLR-normalized values, or another representation.
- If you run the same benchmark on multiple data states, register them as separate dataset slugs so the comparison remains explicit.

## Metadata Columns

Your registered `batch` and `cell_type` keys control which metrics are possible.

```python
adata.obs["batch"] = adata.obs["donor_id"].astype(str)
adata.obs["cell_type"] = adata.obs["manual_annotation"].astype(str)
```

Use stable labels:

- Good `batch` values: `donor_1`, `donor_2`, `10x_v2`, `10x_v3`, `site_a`.
- Good `cell_type` values: `CD4 T`, `B cell`, `monocyte`, `NK`.
- Avoid mixed types, empty strings, and columns with many accidental spellings.

If `cell_type` is missing, mvexp can still run models, but supervised bio-conservation metrics are skipped. If `batch` is missing or has only one value, batch-correction metrics are skipped.

## Prepare RNA AnnData

```python
from pathlib import Path

import scanpy as sc

adata = sc.read_h5ad("notebooks/processed_pbmc.h5ad")

adata.obs["batch"] = adata.obs["donor_id"].astype(str)
adata.obs["cell_type"] = adata.obs["annotation"].astype(str)

# Example PCA-oriented state. Adjust to your study and selected models.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

dataset_dir = Path("store/datasets/pbmc_rna")
data_dir = dataset_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

adata.write_h5ad(data_dir / "rna.h5ad")
```

Create the manifest:

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

with open(dataset_dir / "dataset.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

Register it:

1. Open the **Registry** tab.
2. Expand **Register New Dataset**.
3. Enter `store/datasets/pbmc_rna/dataset.yaml` in **Path to dataset.yaml**.
4. Click **Register Dataset**.
5. Click **Refresh Registry**.

## Prepare RNA+ATAC MuData

```python
from pathlib import Path

import mudata as md

# adata_rna and adata_atac should already be filtered and aligned.
shared_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
adata_rna = adata_rna[shared_cells].copy()
adata_atac = adata_atac[shared_cells].copy()

# MultiVI-style workflows need feature type annotations after concatenation.
adata_rna.var["feature_types"] = "Gene Expression"
adata_atac.var["feature_types"] = "Peaks"

mdata = md.MuData({
    "rna": adata_rna,
    "atac": adata_atac,
})

mdata.obs["batch"] = adata_rna.obs["donor_id"].astype(str)
mdata.obs["cell_type"] = adata_rna.obs["cell_type"].astype(str)

dataset_dir = Path("store/datasets/pbmc_multiome")
data_dir = dataset_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

mdata.write_h5mu(data_dir / "processed.h5mu")
```

Manifest:

```python
import yaml

manifest = {
    "name": "PBMC Multiome RNA+ATAC",
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

with open(dataset_dir / "dataset.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

## Prepare RNA+ADT Data

For TotalVI-style studies, RNA and protein measurements should be aligned by cell. If you keep protein expression in an AnnData `.obsm`, use the convention expected by the model wrapper.

```python
from pathlib import Path

import mudata as md
import pandas as pd

# adata_rna contains gene expression.
# adata_adt contains antibody-derived tag counts or the representation chosen for the study.

shared_cells = adata_rna.obs_names.intersection(adata_adt.obs_names)
adata_rna = adata_rna[shared_cells].copy()
adata_adt = adata_adt[shared_cells].copy()

adata_rna.obsm["protein_expression"] = pd.DataFrame(
    adata_adt.X.toarray() if hasattr(adata_adt.X, "toarray") else adata_adt.X,
    index=adata_adt.obs_names,
    columns=adata_adt.var_names,
)

mdata = md.MuData({
    "rna": adata_rna,
    "adt": adata_adt,
})

mdata.obs["batch"] = adata_rna.obs["donor_id"].astype(str)
mdata.obs["cell_type"] = adata_rna.obs["cell_type"].astype(str)

dataset_dir = Path("store/datasets/citeseq_rna_adt")
data_dir = dataset_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

mdata.write_h5mu(data_dir / "processed.h5mu")
```

Manifest:

```python
import yaml

manifest = {
    "name": "CITE-seq RNA+ADT",
    "omics": ["rna", "adt"],
    "raw_files": {
        "rna": "data/processed.h5mu",
        "adt": "data/processed.h5mu",
    },
    "metadata_keys": {
        "batch": "batch",
        "cell_type": "cell_type",
    },
}

with open(dataset_dir / "dataset.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

## Register Through the GUI

Sequential GUI workflow:

1. Start mvexp and open `http://localhost:8501`.
2. Open the **Registry** tab.
3. Expand **Register New Dataset**.
4. If you already wrote `dataset.yaml`, keep **Build manifest from fields** off and enter the manifest path.
5. If you want the GUI to create the manifest, switch **Build manifest from fields** on and fill in dataset name, omics, file paths, `batch_key`, and `cell_type_key`.
6. Click **Register Dataset**.
7. Click **Refresh Registry**.
8. Confirm the dataset appears as `READY`.

## Read Results Back into Jupyter

Each successful run writes `embeddings.h5` with a dataset named `latent`.

```python
from pathlib import Path

import h5py
import mudata as md
import scanpy as sc

artifact_dir = Path("store/artifacts/run_output/store/artifacts/<artifact-id>")
# Copy the exact path from the Results tab.

with h5py.File(artifact_dir / "embeddings.h5", "r") as f:
    latent = f["latent"][:]

mdata = md.read_h5mu("store/datasets/pbmc_multiome/data/processed.h5mu")
adata = mdata["rna"].copy()
adata.obsm["X_mvexp_multivi"] = latent

sc.pp.neighbors(adata, use_rep="X_mvexp_multivi")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

For `MuData`, attach the embedding to a modality or to a notebook-side AnnData object used for plotting. The important point is that row order must match the cells used by the model run.

## Publishing Checklist

Before writing the paper, keep a copy of the files that make the run reproducible:

- `dataset.yaml` for each dataset;
- the notebook or script that produced the registered `.h5ad` or `.h5mu`;
- `run_manifest.yaml`;
- each successful run's `job_spec.json`;
- `metrics.json`;
- `run.log` and `container.log`;
- `provenance.json` or other provenance artifacts when present in the run directory.

In the Methods section, report:

- matrix state for each modality;
- filtering thresholds;
- normalization and transformation choices;
- HVG or peak selection rules;
- batch and cell-type metadata keys;
- model names and versions;
- random seed;
- metric families used for bio-conservation and batch correction.

Recommended wording:

```text
All integration benchmarks were executed with mvexp. The benchmark plan, including
dataset identifiers, model selections, hyperparameters, random seed, and requested
metrics, is provided as `run_manifest.yaml` in the Supplementary Material. Per-run
runtime instructions and provenance files are provided with the corresponding model
artifacts.
```

## Common Errors

| Symptom | Likely cause | What to do |
|---|---|---|
| Registration fails with missing file | Manifest path points to a file that does not exist. | Check paths relative to `store/datasets/<slug>/`. |
| Batch metrics are skipped | `batch` column is missing or has one unique value. | Inspect `adata.obs["batch"].value_counts()` in Jupyter. |
| Label metrics are skipped | `cell_type` column is missing or misspelled. | Confirm exact `.obs` column names. |
| Returned embedding cannot be plotted | Cell order or object choice does not match the run. | Read the same prepared object used for registration. |

## Cookbook Outline

- **RNA+ATAC Integration for Multiome PBMCs**: prepare paired modalities, register batch and cell-type metadata, compare RNA+ATAC-capable models, and visualize returned embeddings in Scanpy.
- **RNA+ADT CITE-seq Integration**: prepare protein expression alongside RNA, run TotalVI, and compare whether protein-aware latent spaces improve biological label structure.
- **Donor-Aware Atlas Integration**: use donor or site as `batch`, run multiple models across an atlas subset, and report both bio-conservation and batch-correction metrics for publication.
