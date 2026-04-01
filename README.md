# Multi-verse

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Multi-verse is a production-grade framework for the comparative analysis of multimodal single-cell data integration methods. It supports a variety of state-of-the-art models, including MOFA+, MOWGLI, MultiVI, Cobolt, and PCA, providing standardized evaluation metrics and visualizations.

<p align="center">
  <img src="logo.png" alt="Multi-verse Logo" width="200">
</p>

<p align="center">
  <a href="https://github.com/sifrimlab/multi-verse/issues">Report Bug</a> ·
  <a href="https://github.com/sifrimlab/multi-verse/pulls">Add Feature</a>
</p>

<p align="right"><sub><em>Logo generated with the help of ChatGPT.</em></sub></p>

## Overview

Multi-verse simplifies the benchmarking of multimodal integration by providing:

- **Dynamic Routing**: Automatically filters eligible models based on the omics present in your dataset (RNA, ATAC, ADT).
- **Concurrent Orchestration**: Executes multiple models in parallel using isolated Docker containers for maximum performance and stability.
- **Standardized Evaluation**: Integrates `scIB-metrics` to calculate bio-conservation and batch-correction scores (ARI, NMI, Silhouette, etc.).
- **Interactive Setup**: A Streamlit-based wizard to generate configuration files without manual JSON editing.

Supported methods include [MOFA+](https://biofam.github.io/MOFA2/), [Mowgli](https://mowgli.readthedocs.io/en/latest/index.html), [MultiVI](https://docs.scvi-tools.org/en/stable/user_guide/models/multivi.html), Cobolt, and PCA, with scIB metrics and UMAP visualizations for interpretation.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker (for containerized execution)

**Alternative (conda):** You can use a conda environment from `environment.yml` instead of `uv`:

```bash
conda env create -f environment.yml
conda activate multiverse
```

`cmake` may be required for the Louvain dependency to install correctly in some setups.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sifrimlab/multi-verse.git
   cd multi-verse
   ```

2. Install dependencies:

   ```bash
   make install
   ```

   Optional development install (if not using `uv` workflows):

   ```bash
   pip install -e .
   ```

### Running the Setup Wizard

Launch the interactive GUI to generate your configuration:

```bash
make setup
```

### Running the Pipeline

You can run the pipeline with the default Makefile target, directly with Python, or via the Docker orchestrator.

**Option 1: Makefile / `runner.py`**

```bash
make run
# or, with a specific config:
uv run python runner.py config_alldatasets.json
```

If no config file is passed, `runner.py` defaults to `config_alldatasets.json` when invoked that way.

**Option 2: Module entry point**

```bash
python -m multiverse.main config.json
```

**Option 3: Docker orchestrator (concurrent)**

```bash
python -m multiverse.runner.cli --concurrent --input /path/to/data --output /path/to/results --models pca mofa multivi
```

---

## Migrating Legacy Data

Use the migration utility to convert researcher-centric folders into the standardized
`store/datasets/<slug>/data/` layout. The workflow is:

1. Recursively discover directories that contain `.h5ad` / `.h5mu` files.
2. Run `DatasetHeuristics` to infer modalities and likely `batch` / `cell_type` keys.
3. Safely materialize files into the new store (hard-link first, fallback to copy).
4. Write `dataset.yaml` next to `data/`, with comments when multiple metadata alternatives are detected.

Dry-run preview (recommended first):

```bash
python -m multiverse.migrate_data --source /path/to/old --dest /path/to/store --dry-run
```

Run migration:

```bash
python -m multiverse.migrate_data --source /path/to/old --dest /path/to/store
```

The migration is non-destructive and will skip any destination dataset that already
contains a `dataset.yaml` to avoid overwriting manual edits.

---

## Model overview

| Model | Pairing type | Methodology | Hyperparameter evaluation (typical) | scIB metrics |
| --- | --- | --- | --- | --- |
| PCA | Unpaired | Linear dimensionality reduction | Variance explained | Yes |
| MOFA+ | Paired | Variational inference | Variance explained | Yes |
| MultiVI | Paired-guided | Deep generative model | Silhouette (and related) | Yes |
| Mowgli | Paired | Optimal transport + NMF | Optimal transport loss | Yes |
| Cobolt | Paired | Multi-omics joint embedding | Model-specific | Yes |

---

## Configuration reference

The system uses a JSON configuration file. Primary keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `batch_key` | `string` | **Required.** The key in `.obs` identifying experimental batches. |
| `cell_type_key` | `string` | Optional. The key in `.obs` identifying ground-truth cell types. |
| `random_seed` | `int` | Seed for reproducibility (default: 42). |
| `output_dir` | `string` | Path where results and logs will be saved. |
| `data` | `object` | Mapping of dataset names to file paths and per-modality settings. |
| `model` | `object` | Models to run (e.g. `pca`, `mofa`, `multivi`, `mowgli`, `cobolt`) and their hyperparameters. |
| `_run_user_params` | `bool` | Whether to run models with the specified parameters. |
| `_run_gridsearch` | `bool` | Enable hyperparameter search using each model’s `grid_search_params`. |
| `preprocess_params` | `object` | Optional filtering/normalization settings for RNA, ATAC, and ADT. |
| `training` | `object` | Optional training-related options. |

### Datasets (`data`)

Each dataset entry includes:

- `data_path`: Directory (or file path, depending on layout) where modality files live.
- `rna`, `atac`, `adt` (optional): Per-modality blocks with:
  - `file_name`: Data file name.
  - `is_preprocessed`: Whether data are already preprocessed.
  - `annotation`: Optional key for cell types or other labels used in evaluation/plots.

Example `data` entry:

```json
"dataset_1": {
  "data_path": "data/pbmc.h5mu",
  "rna": { "file_name": "rna.h5ad", "is_preprocessed": false }
}
```

The repo ships example configs such as `dataset_Pbmc10k` (RNA + ATAC). Example datasets referenced in older tutorials:

- **dataset_Pbmc10k** — [Download (Google Drive)](https://drive.google.com/drive/u/0/folders/1uq6UJFaCqcrV7XjAiNmfdptKW0BfL0Ha): RNA and ATAC from ~10k PBMCs; includes cell type annotations.
- **dataset_TEA** — Same folder: RNA, ATAC, and ADT from a leukopak sample; annotation may be absent but useful for multi-modal tests.

### Models (`model`)

Each enabled model is a key under `model` (e.g. `pca`, `mofa`, `multivi`, `mowgli`, `cobolt`). Common fields:

- `device`: `cpu` or `cuda:<index>`.
- `umap_random_state`, `umap_color_type`, `umap_use_representation` (where applicable).
- `grid_search_params`: Hyperparameter grids used when `_run_gridsearch` is true.

Model-specific hyperparameters vary; see `config_alldatasets.json` for full examples.

### Preprocessing (`preprocess_params`)

Optional structure for RNA, ATAC, and ADT filtering and normalization (e.g. min/max genes by counts, normalization targets, ADT per-cell normalization). The default device for preprocessing can be aligned with your `device` settings in each model block.

---

## Results format

### Grid search

`_run_gridsearch` and per-model `grid_search_params` are part of the configuration schema. The main workflow may still log that grid search is skipped while that path is fully wired; see `multiverse/main.py` for current behavior. When grid search is fully executed, outputs are typically organized under your configured `output_dir` (historically tutorials used `./outputs/gridsearch_output/` for best-run artifacts and console summaries).

### Evaluation

Metrics come from [scIB-metrics](https://scib-metrics.readthedocs.io/en/stable/) on latent embeddings. Per-dataset results are written under `output_dir/<dataset_name>/` (e.g. `evaluation_metrics.json`). Aggregated summaries may also appear as `results.json` at the `output_dir` root when the aggregation step runs.

Commonly reported metrics include:

- **ARI** — Clustering agreement with known annotations.
- **NMI** — Normalized mutual information between clusters and labels.
- **Silhouette** — Separation quality of clusters.
- **Graph connectivity** — Batch mixing / integration.
- **Isolated labels ASW** — How well isolated populations are preserved after integration.

---

## Developer guide

### Adding a new model

1. **Implement the wrapper**: Add a class in `multiverse/models/` inheriting from `ModelFactory`.
2. **Update the registry**: Add metadata in `model_registry.json`.

   ```json
   {
     "name": "new_model",
     "docker_image": "multiverse-new_model:latest",
     "supported_omics": ["rna", "atac"]
   }
   ```

3. **Containerize**: Add a Dockerfile under `containers/` and build before running the orchestrator.

### Master–worker flow

1. **Master process**: Validates config, loads the model registry, and checks dataset omics.
2. **Dynamic routing**: Drops models incompatible with the dataset’s modalities.
3. **Preparation**: Pulls or builds required Docker images concurrently.
4. **Execution**: Worker containers mount input data **read-only**.
5. **Aggregation**: Collects outputs and builds the evaluation summary.

---

## Troubleshooting

### Docker build issues

1. **Check Docker**

   ```bash
   docker --version
   docker info
   ```

2. **Build manually**

   ```bash
   docker build -t multiverse:test .
   ```

3. **Inspect logs** for missing dependencies, network errors, or Dockerfile issues.

4. **Shell into a container**

   ```bash
   docker run -it multiverse:test /bin/bash
   ```

5. **Logs**

   ```bash
   docker logs <container_id>
   ```

6. **Dependencies**: Confirm requirements match your base image and `requirements.txt` / lockfile.

7. **Clean rebuild**

   ```bash
   docker system prune -a
   docker build --no-cache -t multiverse:test .
   ```

### Common issues

- **Out of memory**: Increase Docker memory limits (e.g. Docker Desktop).
- **Network timeouts**: Check proxies and connectivity for image pulls.
- **Permission errors**: Ensure the Docker daemon is running and your user can access it.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes with clear messages.
4. Push and open a pull request.

---

## Contact

**Project:** [github.com/sifrimlab/multi-verse](https://github.com/sifrimlab/multi-verse)

### Contributors

Developed as part of the Integrated Bioinformatics Project (B-KUL-I0U20A), Faculty of Bioscience Engineering, KU Leuven.

**Authors**

- [Yuxin Qiu](https://github.com/yuxin0924)
- [Thi Hanh Nguyen Ly](https://github.com/HannahLy1204)
- [Zuzanna Olga Bednarska](https://github.com/ZOBednar)

**Supervisors:** Anis Ismail, Lorenzo Venturelli  
**Promotor:** Prof. Alejandro Sifrim  
**Course coordinator:** Prof. Vera van Noort

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
