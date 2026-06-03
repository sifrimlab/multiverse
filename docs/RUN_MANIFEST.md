# Run Manifest Reference

The run manifest is the recipe for a benchmark. It is a YAML document at the repository root (conventionally `run_manifest.yaml`) that the GUI writes from the **Configure** tab and that mvd consumes in the **Run** tab or via the CLI:

```bash
multiverse run --manifest run_manifest.yaml --output ./results
```

A manifest is the single artifact a reader needs in order to reproduce a benchmark. It travels alongside each artifact directory as `run_manifest.yaml` and should be included with publication supplementary materials.

## Top-Level Structure

```yaml
globals:
  experiment_name: <slug>
  random_seed: 42
  run_user_params: true
  run_gridsearch: false
  metrics:                  # optional
    bio_conservation: [ari, nmi, silhouette_label]
    batch_correction: [silhouette_batch, ilisi, kbet]
jobs:
  - dataset_slug: <slug>
    model_name: <DisplayName>
    model_params:
      <hyperparameter>: <value>
```

## `globals` Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `experiment_name` | string (alphanum + `-`) | yes | Used as the run grouping label and, when projection sync is enabled, the MLflow experiment name. |
| `random_seed` | integer | yes | Propagated into every `job_spec.json`. Containers must apply this seed before any stochastic call. |
| `run_user_params` | boolean | yes | When `true`, the runner uses the parameters in each `jobs[].model_params` block. When `false`, the runner uses each model's registered defaults. |
| `run_gridsearch` | boolean | yes | When `true`, each job becomes an Optuna study with the parameter distributions defined in the model's hyperparameter schema. |
| `skip_completed` | boolean | no | Opt-in resume (default `false`). When `true`, a job is skipped if its canonical logical run already reached `ARTIFACT_SUCCESS` in the mvd state for the chosen output directory. Overridden by the CLI `--skip-completed` flag or by an explicit GUI checkbox change; an untouched GUI checkbox honors this manifest value. The legacy `runs` table is never consulted. See [Runner & Orchestration -> Resuming Completed Work](RUNNER.md#resuming-completed-work-skip_completed). |
| `metrics.bio_conservation` | list[string] | no | Limits which bio-conservation metrics are computed; default is all metrics whose preconditions are satisfied. |
| `metrics.batch_correction` | list[string] | no | Same, for batch-correction metrics. |

## `jobs` Fields

Each list entry is one dataset × model run.

| Field | Type | Required | Description |
|---|---|---|---|
| `dataset_slug` | string | yes | Slug of a dataset registered in the `datasets` table. |
| `model_name` | string | yes | Display name of a registered model (`PCA`, `MOFA`, `MultiVI`, `Mowgli`, `Cobolt`, `TotalVI`). |
| `model_params` | object | no when `run_user_params: false` | Key-value parameter map validated against the model's JSON schema. |
| `gpu` | boolean | no | Request GPU access for this job. GPU is **opt-in** and defaults to `false`; it is honored only when a GPU is actually available on the host (otherwise the container runs on CPU). |
| `mem_limit` | string | no | Docker memory limit for this job (e.g. `"32g"`). Defaults to the model's `resources.memory_limit`. |
| `preprocessing` | object | no | Per-run preprocessing overrides (`n_top_genes`, `normalization_target_sum`, `log_normalization`, `scale`). Written into the container's `job_spec.json` and merged over the model's built-in defaults; omitted fields fall back to those defaults. |

When `run_gridsearch: true`, `model_params` values may be search-space specifications (e.g. `{distribution: loguniform, low: 1e-4, high: 1e-1}`) rather than concrete scalars. The Configure tab renders the correct controls based on the schema.

> **GPU access (issue #30):** GPU is never requested implicitly. A job must set `gpu: true` (or the model's `resources.gpu` default, surfaced via the GUI "Enable GPU" toggle) to receive a Docker `device_requests` / Apptainer `--nv` allocation, and only when a GPU is present. Simple-mode manifests use `model.gpu` on each job instead.

## Validation

The runner parses the manifest into a `ParsedManifest` dataclass with an explicit error list. Validation runs before any container starts and surfaces:

- unknown dataset slugs;
- unknown or version-mismatched model names;
- omics not supported by the requested model;
- missing `batch_key` / `cell_type_key` when the requested metrics need them;
- `model_params` values that fail the hyperparameter schema.

Errors are reported in aggregate so a malformed manifest is reported once, not job-by-job.

## Example

```yaml
globals:
  experiment_name: pbmc-baselines
  random_seed: 42
  run_user_params: true
  run_gridsearch: false
jobs:
  - dataset_slug: pbmc10k
    model_name: PCA
    model_params:
      n_components: 20
      device: cpu
      umap_random_state: 42
      umap_color_type: cell_type
  - dataset_slug: pbmc10k
    model_name: Cobolt
    gpu: true
    mem_limit: "32g"
    preprocessing:
      n_top_genes: 2000
      log_normalization: true
    model_params:
      device: cuda:0
      latent_dimensions: 10
      learning_rate: 0.001
      num_epochs: 50
      umap_random_state: 42
      umap_color_type: cell_type
      random_state: 42
```

## Lifecycle

1. The GUI Configure tab writes `run_manifest.yaml`.
2. The Run tab (or the CLI) reads it and validates it.
3. The orchestrator launches jobs and copies the manifest verbatim into each run's artifact directory as `run_manifest.yaml`.
4. The same content is logged as an MLflow artifact attached to the parent run.

Because each run carries the manifest, the recipe is recoverable from any single artifact directory or MLflow entry — the database is not the only source.
