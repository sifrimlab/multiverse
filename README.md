# Multi-verse

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Benchmark any multimodal single-cell integration model — in any language — with one YAML file.**

Multi-verse is a language-agnostic MLOps platform for multimodal single-cell integration.
Write a manifest, register your dataset, and the platform handles parallel container dispatch,
seed injection, automatic HPO via Optuna, MLflow provenance, and an end-of-run comparison table.
Built-in models cover RNA, ATAC, and protein modalities; adding an R or Julia model takes 15 minutes.

| What you get | How |
|---|---|
| Reproducible results | Seeds flow from manifest → `job_spec.json` → every framework call |
| Fair comparison | All models run in parallel, isolated containers with the same data mount |
| Automated HPO | Optuna sweeps across any hyperparameter defined in the model's JSON schema |
| Full provenance | Every artifact dir contains the exact manifest that produced it |
| No framework lock-in | Container I/O contract works for Python, R, Julia, or shell scripts |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker

### 1) Initialize

```bash
make init
make install
```

### 2) Optional GUI Setup

```bash
make setup
```

### 3) Register Data and Models

For model registration, make sure each model package includes:
- model script/module entrypoint
- `store/models/<slug>/container/Dockerfile`
- `store/models/<slug>/container/environment.yml`
- `schemas/models/<slug>.hyperparameters.schema.json`
- `store/models/<slug>/model.yaml`

```bash
make register slug=<dataset-slug>
make register-model slug=<model-slug>
# or register all built-in models at once
make register-models
```

### 4) Run Benchmark

```bash
make benchmark MANIFEST=run_manifest.yaml OUTPUT_DIR=./results
```

## Architecture

```
dataset.yaml + model.yaml
        │
        ▼
 SQLite Registry (datasets, models, runs)
        │
        ▼
 Orchestrator Planner
        │
        ├─ Pre-flight Validation Gate
        │   ├─ Omics compatibility check (skip incompatible jobs)
        │   ├─ batch_key presence check (skip if absent from obs)
        │   ├─ cell_type_key warning (warn, don't skip)
        │   └─ Single-batch warning (warn, don't skip)
        │
        ├─ Parallel Image Build/Pull (all images built before any job starts)
        │
        ├─ Parallel Container Execution (asyncio.gather)
        │   ├─ /input/data.h5mu  (read-only mount)
        │   ├─ /output/job_spec.json  (seed, hyperparameters, metrics config)
        │   └─ /output/{embeddings,metrics,umap}
        │
        ├─ Workspace Promotion (exit_code == 0 → atomic move to artifacts/)
        │
        └─ End-of-Run Summary Table (success / failed / skipped per job + warnings)
```

## Configuration Reference

### `run_manifest.yaml` globals

```yaml
globals:
  experiment_name: benchmark_run
  random_seed: 42              # Seed passed to ALL model containers
  metrics:                     # Optional: which scib-metrics benchmarks to run
    bio_conservation:
      - silhouette_label
      - nmi_ari_cluster_labels_leiden
    batch_correction:
      - graph_connectivity
      - ilisi_knn
```

### Per-job fields

```yaml
jobs:
  - dataset_slug: pbmc10k
    model_name: PCA
    model_params:
      n_components: 50
      device: cpu
    metrics:                   # Optional: override globals for this job
      model_metrics:           # Controls which per-model metrics are computed
        - total_variance
```

### Dataset registration (`dataset.yaml`)

```yaml
slug: pbmc10k
name: PBMC 10k
path: store/datasets/pbmc10k/data.h5mu
omics_available: [rna, atac]
batch_key: batch              # Must exist in dataset.obs; used for batch-correction metrics
cell_type_key: cell_type      # Used for supervised metrics; absence triggers a warning
```

## Pre-flight Validation

Before any container starts, the orchestrator validates each planned job:

| Check | Outcome |
|---|---|
| Dataset missing required omics for the model | Job **skipped** with explanation |
| `batch_key` absent from dataset `.obs` | Job **skipped** with explanation |
| `cell_type_key` absent from dataset `.obs` | **Warning** only — job proceeds, supervised metrics skipped |
| Only 1 unique batch value | **Warning** only — job proceeds, batch-correction metrics skipped |

Skipped and warned jobs appear in the end-of-run summary table.

## Reproducibility

The `random_seed` in `run_manifest.yaml` (or `--seed` on the CLI) is written into `job_spec.json` and consumed by every model container:

| Model | Seed calls |
|---|---|
| PCA | `random.seed`, `numpy.random.seed` |
| MultiVI | `scvi.settings.seed` (covers PyTorch + NumPy + random) |
| TotalVI | `scvi.settings.seed` |
| MOFA | `random.seed`, `numpy.random.seed` |
| Mowgli | `random.seed`, `numpy.random.seed`, `torch.manual_seed` |
| Cobolt | `random.seed`, `numpy.random.seed`, `torch.manual_seed` |

UMAP is seeded separately via the `umap_random_state` model parameter.

## Metrics

Each model writes `metrics.json` on completion:

| Model | Default metrics |
|---|---|
| PCA | `total_variance` |
| MultiVI | `silhouette_score` (if cell type labels available) |
| TotalVI | `elbo_train`, `reconstruction_loss_train` |
| MOFA | `total_variance` |
| Mowgli | `ot_loss` |
| Cobolt | `loss` |

To restrict which metrics are computed, add `metrics.model_metrics` to a job in `run_manifest.yaml`.

Full multi-model benchmarking (scib-metrics: silhouette, NMI/ARI, iLISI, kBET, etc.) runs via the evaluation container after all model jobs complete.

## Design Principles

- **Contract over code:** any language/runtime is allowed (Python, R, Julia, etc.) if the container obeys the mount contract.
- **Data-centric and model-centric workflows:** run 1 dataset against N models, or 1 model against N datasets.
- **Immutable artifacts:** only successful runs are promoted.
- **Fail fast on data issues:** incompatible jobs are skipped before containers start, not discovered inside them.
- **Resumable sweeps:** Optuna uses persistent SQLite study storage.
- **Schema-driven UX:** GUI hyperparameter fields are generated from each model's JSON schema.

## Key Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — sequence diagram, DB schema, container contract, concurrency model
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — model onboarding tutorial (15-minute walkthrough for any language)
- [docs/DATA_REGISTRATION.md](docs/DATA_REGISTRATION.md)
- [docs/MODEL_REGISTRATION.md](docs/MODEL_REGISTRATION.md)
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md)
- [docs/MODEL_CONTAINERS.md](docs/MODEL_CONTAINERS.md)
- [CHANGELOG.md](CHANGELOG.md)

## License

Distributed under the MIT License. See `LICENSE` for details.
