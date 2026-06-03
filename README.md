# mvexp

**Reproducible benchmarking for multimodal single-cell integration, without making bioinformaticians become infrastructure engineers.**

mvexp is an MLOps platform for academic biological integration studies. You bring the scientific objects you already use in notebooks: `AnnData`, `MuData`, batch annotations, cell-type labels, and model questions. mvexp handles the repetitive infrastructure around registration, model execution, parameter tracking, artifacts, Optuna sweeps, and MLflow comparison.

The goal is simple: make it easier to run a defensible benchmark and easier to explain exactly what you did in a paper.

## Who This Is For

mvexp is designed for researchers who are comfortable with Scanpy, Seurat, MuData, and Jupyter, but do not want every benchmark to become a Docker and orchestration project.

You should be able to answer scientific questions such as:

- Which integration model best preserves my annotated cell populations?
- Does batch correction remove donor or chemistry effects without erasing biology?
- Are conclusions stable across random seeds and hyperparameters?
- Can I attach enough provenance for a reviewer to reproduce my benchmark?

## What mvexp Handles for You

| You focus on | mvexp handles |
|---|---|
| Biological question and dataset curation | Dataset registration and compatibility checks |
| Batch and cell-type metadata | Metric eligibility and clear warnings |
| Model choice and hyperparameters | Safe, parallel execution without hand-running containers |
| Comparing embeddings and metrics | Results tables, artifacts, MLflow tracking, and Optuna sweeps |
| Writing a reproducible Methods section | `run_manifest.yaml`, `job_spec.json`, metrics, logs, and provenance artifacts |

## How the Workflow Feels

1. Prepare your `AnnData` or `MuData` object in Jupyter.
2. Save it under `store/datasets/<dataset-slug>/data/`.
3. Open the Streamlit GUI.
4. Register the dataset in the Registry tab.
5. Choose dataset x model pairs in Job Builder.
6. Set parameters in the Parameters tab.
7. Launch the run in Run.
8. Review metrics, embeddings, logs, and artifacts in Results and MLflow.
9. Bring the selected `embeddings.h5` back into Jupyter for figures and downstream analysis.

## Quick Start

Prerequisites:

- Python 3.12+
- `uv`
- Docker with Compose v2

Set up the platform from a source checkout:

```bash
make bootstrap      # install dev deps, initialize the registry, register built-in models
make services-up    # optional: MLflow on :25000 and Optuna Dashboard on :28080
make setup          # optional: install GUI and ML model wrapper extras
make gui            # launch Streamlit on :28501
```

Then open the GUI at `http://localhost:28501` (or the port in `.env`). For a headless run, generate or edit `run_manifest.yaml` and use:

```bash
uv run multiverse run --manifest run_manifest.yaml --output store/artifacts/run_output
```

On HPC clusters with Slurm and Apptainer, use the Slurm path instead of Docker:

```bash
uv run multiverse slurm-submit \
  --model-slug pca \
  --image-sif /scratch/images/multiverse-pca-1.0.0.sif \
  --image-digest sha256:<oci-digest> \
  --params-json '{"n_components": 20}' \
  --output store/artifacts/run_output
```

See [Runner & Orchestration](docs/RUNNER.md) for the full Slurm workflow including dual-digest artifact manifests.

In the GUI:

1. Open the **Registry** tab.
2. Expand **Register New Dataset**.
3. Either provide an existing `dataset.yaml`, or switch on **Build manifest from fields**.
4. Click **Register Dataset**.
5. Open **Job Builder** and select compatible dataset x model pairs.
6. Open **Parameters** and set model hyperparameters.
7. Open **Run** and click **Launch Run**.
8. Open **Results** to inspect completed runs and artifact paths.

For a full guided tutorial, see [Getting Started](docs/GETTING_STARTED.md).

## The Jupyter Bridge

Most users should keep doing their exploratory work in notebooks. mvexp is meant to sit between notebook curation and notebook interpretation.

### Save a MuData Object for mvexp

```python
from pathlib import Path

import mudata as md

# Assume you already created these in Scanpy:
# adata_rna  : AnnData with RNA counts/features
# adata_atac : AnnData with ATAC counts/features

dataset_dir = Path("store/datasets/pbmc_rna_atac")
data_dir = dataset_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

mdata = md.MuData({
    "rna": adata_rna,
    "atac": adata_atac,
})

# Shared sample/cell metadata used by evaluation.
mdata.obs["batch"] = adata_rna.obs["batch"].astype(str)
mdata.obs["cell_type"] = adata_rna.obs["cell_type"].astype(str)

mdata.write_h5mu(data_dir / "processed.h5mu")
```

Create the dataset manifest next to the data:

```python
import yaml

manifest = {
    "name": "PBMC RNA+ATAC",
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

Then register `store/datasets/pbmc_rna_atac/dataset.yaml` in the GUI Registry tab.

### Read an Embedding Back into Jupyter

After a successful run, copy the artifact path from the Results tab. The embedding file contains one HDF5 dataset named `latent`.

```python
from pathlib import Path

import h5py
import mudata as md
import scanpy as sc

artifact_dir = Path("store/artifacts/run_output/store/artifacts/<artifact-id>")
# In practice, copy the exact artifact directory from the Results tab.

with h5py.File(artifact_dir / "embeddings.h5", "r") as f:
    latent = f["latent"][:]

mdata = md.read_h5mu("store/datasets/pbmc_rna_atac/data/processed.h5mu")
adata = mdata["rna"].copy()
adata.obsm["X_mvexp_pca"] = latent

sc.pp.neighbors(adata, use_rep="X_mvexp_pca")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

See [Data Preparation](docs/DATA_PREPARATION.md) for more detailed examples for `AnnData`, `MuData`, RNA+ATAC, and RNA+ADT studies.

## Data State: Biological Assumptions

mvexp does not decide what counts as appropriate biological preprocessing for your study. It makes your choices explicit and reproducible.

Recommended expectations:

- Use matrices in the state expected by the selected model. Count-based probabilistic models generally expect count-like input; PCA-style baselines are often run on normalized/log-transformed features.
- Keep raw counts available when possible, especially for RNA, ATAC, and ADT models that assume count data.
- Preserve feature annotations such as `feature_types` when models need to distinguish genes, peaks, and proteins.
- Provide a `batch`-like column for technical or donor effects you want assessed.
- Provide a `cell_type`-like column when you want supervised bio-conservation metrics such as ARI, NMI, label silhouette, or cLISI.
- Store metadata columns as categorical or string-like values, not mixed Python objects.
- Record HVG selection and filtering choices in your notebook or supplementary methods.

The platform checks whether metadata columns exist and whether batch metrics are meaningful. It does not silently invent biological labels.

## Publishing and Reproducibility

Every run is designed to leave a Methods trail:

- `run_manifest.yaml` records the selected datasets, models, parameters, metrics, experiment name, and seed.
- `job_spec.json` records the exact runtime instructions passed to each model container.
- `metrics.json` records model-level metrics and histories when available.
- `embeddings.h5` records the latent representation used for downstream evaluation.
- logs and provenance artifacts document what ran and where outputs were written.

For a paper, include `run_manifest.yaml` and `provenance.json` when it is present in the run directory as Supplementary Material. Together with the archived input data, these files are the reproducibility contract for the benchmark: they record what was run, with which parameters, seed, metrics, and artifacts. In your Methods section, describe the dataset state, metadata keys, selected models, random seed, and metric families.

## Documentation Map

The docs follow Diátaxis:

| Type | Document | Purpose |
|---|---|---|
| Tutorial | [Getting Started](docs/GETTING_STARTED.md) | First complete run from notebook object to results. |
| How-To | [Data Preparation](docs/DATA_PREPARATION.md) | Practical recipes for making data acceptable to mvexp. |
| How-To | [Runner & Orchestration](docs/RUNNER.md) | Docker and Slurm execution paths, `--accept-degraded`, asset registry migration. |
| Reference | [Models Glossary](docs/reference/MODELS_GLOSSARY.md) | Built-in model assumptions and hyperparameters. |
| Reference | [Evaluation Metrics](docs/reference/EVALUATION_METRICS.md) | Bio-conservation and batch-correction metric definitions. |
| Explanation | [Architecture](docs/ARCHITECTURE.md) | How the platform works internally, including the dual-SQLite split and sole-writer invariant. |
| Explanation | [Developer Guide](docs/DEVELOPER_GUIDE.md) | Codebase conventions, test suite, and core boundaries. |

## Cookbook Preview

- **RNA+ATAC Integration for PBMC Multiome**: prepare paired RNA and ATAC modalities, compare MultiVI, MOFA, Mowgli, Cobolt, and PCA, then inspect whether immune cell labels remain separated.
- **CITE-seq RNA+ADT Protein Integration**: prepare RNA and antibody-derived tags for TotalVI, evaluate protein-aware latent structure, and bring embeddings back into Scanpy.
- **Cross-Donor Batch Correction in an Atlas Subset**: register donor metadata as the batch key, compare models by bio-conservation and batch-correction metrics, and produce a reproducible Methods bundle.

## How to Cite mvexp

Until a formal paper or DOI is available, cite the repository or archived release used for the benchmark. Include the mvexp version or commit hash, `run_manifest.yaml`, and run provenance artifacts with Supplementary Material.

Suggested wording:

```text
Integration benchmarks were executed with mvexp (version/commit: <commit>). The full
benchmark recipe, including datasets, models, hyperparameters, random seed, and requested
metrics, is provided as Supplementary File X (`run_manifest.yaml`). Per-run provenance and
model artifacts are archived with the analysis.
```

## License

Distributed under the MIT License. See `LICENSE` for details.
