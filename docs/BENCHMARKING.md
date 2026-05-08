# Benchmarking Workflows

This document explains how to benchmark with the manifest-driven orchestrator.

## Core Concept

Benchmarking is defined in `run_manifest.yaml` and executed by the registry-aware CLI.

Run command:

```bash
make benchmark MANIFEST=run_manifest.yaml OUTPUT_DIR=./results
```

## Manifest Structure

Use two layers:

- **Globals** (optional): shared settings for tracking/tuning/runtime.
- **Jobs** (required): dataset/model execution units.
- **Model Params** (optional): passed per job as `model_params`; can be authored manually or generated from GUI schema forms.

Example:

```yaml
manifest_version: "1.0"
globals:
  experiment_name: "pbmc-benchmark"
  seed: 42
  mlflow_tracking_uri: "file:./mlruns"
  mlflow_experiment_name: "pbmc-benchmark"

jobs:
  - dataset_id: pbmc10k
    models: [pca, mofa, multivi]
    mode: run
  - dataset_id: teaseq
    models: [totalvi]
    mode: run
```

## Workflow 1: Data-Centric (1 Dataset vs N Models)

Use one `dataset_id` and list multiple `models`.

```yaml
jobs:
  - dataset_id: pbmc10k
    models: [pca, mofa, multivi, mowgli, cobolt, totalvi]
    mode: run
```

Use this when you want broad model comparison on one biological dataset.

## Workflow 2: Model-Centric (1 Model vs N Datasets)

Repeat jobs across datasets with one model.

```yaml
jobs:
  - dataset_id: pbmc10k
    models: [multivi]
    mode: run
  - dataset_id: teaseq
    models: [multivi]
    mode: run
```

Use this when you want to stress-test one model’s robustness across cohorts/technologies.

## Enabling Optuna Sweeps

Set `mode: sweep` and include sweep config:

```yaml
jobs:
  - dataset_id: pbmc10k
    models: [pca]
    mode: sweep
    optimize_metric: total_variance
    direction: maximize
    n_trials: 20
    study_storage: sqlite:///optuna.db
    search_space:
      n_components:
        type: int
        low: 10
        high: 100
      solver:
        type: categorical
        choices: [auto, full]
      learning_rate:
        type: loguniform
        low: 0.0001
        high: 0.1
```

### Why `study_storage` matters

Use SQLite-backed storage so interrupted sweeps can resume after reboot:

- `sqlite:///optuna.db`

## Viewing Results in MLflow

If tracking is configured, successful runs are proxied to MLflow by the orchestrator:

- hyperparameters from `job_spec.json`
- metrics from `metrics.json`
- full promoted artifact directory as run artifacts

Start UI:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## GUI Hyperparameter Forms

The setup wizard reads each selected model's `hyperparameters_schema` from the registry and builds typed input controls automatically:

- `enum` -> select box
- `integer`/`number` -> numeric input (honors min/max/default when provided)
- `boolean` -> checkbox
- other types -> text input

If a model schema is missing or invalid, the GUI falls back to raw JSON input for that dataset-model pair.

