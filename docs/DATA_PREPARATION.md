# Data Preparation

This how-to explains how to prepare biological data for multiverse.

Multiverse expects a well-formed `MuData` file, plus explicit metadata keys that say which column represents batch and which column represents biological labels.

## Data State: What multiverse Assumes

Multiverse does not perform scientific preprocessing decisions for you. It preserves and records the choices you make before registration.

| Requirement | Practical guidance |
|---|---|
| Cells are observations | Cell barcodes should be in `.obs_names`. |
| Features are variables | Genes, peaks, proteins, etc. should be in `.var_names`. |
| Modalities are explicit | Use `MuData` modalities such as `rna`, `atac`, and `adt` for multimodal studies. |
| `batch_key` metadata is present | Use a string or categorical `.obs` column for donor, chemistry, lane, site, or another nuisance source. If not available, we will assign the same default value for all samples.|
| `cell_type_key` metadata is present when available | Use a string or categorical `.obs` column. |
| Preprocessing | Store the matrix state as raw counts, perform the processing expected by the selected models in the implementation through the GUI/manifest file, and document normalization, filtering, and HVG selection. |

## Raw Counts, Normalized Data, and HVGs

Different models make different assumptions. Multiverse records what you run, but it does not convert one scientific data state into another without your knowledge.

Recommended practice:

- Keep raw counts in a layer or adjacent file when possible.
- For PCA baselines, normalized and log-transformed matrices with HVG selection are often appropriate.
- For probabilistic count models, preserve count-like input unless the model documentation says otherwise.

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

If `cell_type` or `batch` is missing, Multiverse can still run models, but the evaluation metrics pipeline cannot run without having both available. To circumvent this limitation in the scib-metrics source code, we assign random labels for the samples for each of the missing keys (for now). **Therefore, in case a label is misisng, the results shown for that missing label might be misleading.**

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


## Register Through the GUI

Sequential GUI workflow:

1. Start multiverse and open `http://localhost:28501`.
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
adata.obsm["X_multiverse_multivi"] = latent

sc.pp.neighbors(adata, use_rep="X_multiverse_multivi")
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
All integration benchmarks were executed with multiverse. The benchmark plan, including
dataset identifiers, model selections, hyperparameters, random seed, and requested
metrics, is provided as `run_manifest.yaml` in the Supplementary Material. Per-run
runtime instructions and provenance files are provided with the corresponding model
artifacts.
```
