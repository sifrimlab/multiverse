# Streamlit GUI

The Streamlit application at `multiverse/gui.py` is the primary entry point for researchers. It exposes the registry, the benchmark planner, the runner, and the results browser as a five-tab layout. Headless execution via `python -m multiverse.runner.cli run --manifest run_manifest.yaml` remains supported for scripted pipelines, but the GUI is the path tested against the day-to-day workflow.

## Layout

The application is a single-page Streamlit app driven by query-parameter routing (`multiverse/gui_navigation.py`). Five canonical tabs are exposed:

| Tab | Query parameter | Purpose |
|---|---|---|
| Registry | `?tab=registry` | Inspect, register, and refresh datasets and models. |
| Configure | `?tab=configure` | Pick dataset × model pairs and set hyperparameters. |
| Run | `?tab=run` | Launch and supervise a benchmark execution. |
| Results | `?tab=results` | Browse completed runs, metrics, logs, and artifacts. |
| Analysis | `?tab=analysis` | Embedded MLflow and Optuna views for cross-run analysis. |

For backward compatibility the navigation layer rewrites the older URLs (`jobs`, `params`, `execute`, `mlflow`, `optuna`, `settings`) to their current counterparts. Bookmarks from earlier releases continue to work.

## Registry

The Registry tab is the only point where dataset and model rows are created or refreshed against the SQLite database (`mvexp_state.db`).

- **Datasets table** lists every registered dataset with its slug, omics, batch key, cell-type key, and status (`READY` / `STALE`).
- **Register New Dataset** accepts either an existing `dataset.yaml` path or a "Build manifest from fields" form. Either path writes the manifest to `store/datasets/<slug>/dataset.yaml`, hashes it, and upserts the row.
- **Models table** lists registered models with their version, contract version, supported omics, and Docker image tag.
- **Register / Refresh Model** re-reads `store/models/<slug>/model.yaml` and re-validates the hyperparameter schema.

Status `STALE` indicates the manifest on disk differs from the hash stored in the database; clicking refresh re-syncs the row.

## Configure

Configure is the merger of what older releases called *Job Builder* and *Parameters*. It is organized as a single page so that the dataset-by-model matrix and the parameter forms stay in sync.

- The compatibility matrix is computed by `multiverse.registry.generate_compatibility_matrix`. A cell is `Compatible` when all model-required omics are present; `Partial` when at least one is present; `Incompatible` otherwise. Only `Compatible` pairs are selectable.
- Selected pairs become rows in the run manifest. For each row a parameter form is rendered from the model's hyperparameter JSON schema (`schemas/models/<slug>.hyperparameters.schema.json`).
- Sweep controls appear next to each parameter when the schema marks the field as sweepable; toggling them switches the field to an Optuna distribution (`uniform`, `loguniform`, `int`, `categorical`).
- The page emits a `run_manifest.yaml` preview and persists it to disk on demand.

## Run

The Run tab invokes the same orchestrator that backs the CLI. It streams JSON events from the runner over stderr and renders them as a live status table.

- **Manifest path** defaults to `run_manifest.yaml` at the repository root.
- **Output directory** defaults to `store/artifacts/<experiment_name>/`.
- **Launch Run** triggers `multiverse.runner.cli run`. Jobs are pre-flighted (omics, batch key, cell-type key), then images are pulled or built in parallel, then container jobs run under an asyncio semaphore that respects host RAM and CPU.
- The status table shows per-job state: `PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `SKIPPED`. A skipped job is informative, not an error — it usually means a metric requirement was not satisfied.

## Results

The Results tab reads from the `runs` table and from the artifact tree on disk.

- Filter by experiment, dataset, model, and status.
- Selecting a run opens the artifact browser (`multiverse/gui_artifacts.py`) which shows `run_manifest.yaml`, `job_spec.json`, `metrics.json`, `embeddings.h5`, `umap.png`, and `container.log`.
- The metrics table is grouped by metric family (bio-conservation, batch-correction, model history).
- Artifact paths can be copied directly into a Jupyter session for downstream analysis.

## Analysis

The Analysis tab embeds the MLflow and Optuna dashboards that `make services-up` launches in Docker Compose.

- **MLflow** at `http://localhost:5000` is the canonical home for cross-run comparison, parameter logging, and metric histories.
- **Optuna Dashboard** at `http://localhost:8080` visualizes sweep trials and parameter importance for runs launched with `run_gridsearch: true`.

The Analysis tab is iframe-based and assumes the services are reachable from the browser; if the panels are blank, verify with `docker compose ps` or `make status`.

## Common Issues

| Symptom | Likely cause | What to do |
|---|---|---|
| Dataset missing from Configure | Registry cache stale. | Open Registry → **Refresh Registry**. |
| Model row is `Incompatible` | Dataset omics do not match model requirements. | Choose another model or re-register the dataset with the correct modality. |
| Parameter controls missing | Model schema not registered or invalid. | Re-run `make register-model slug=<slug>`. |
| Launch fails before container start | Manifest references stale dataset/model rows. | Regenerate the manifest from Configure. |
| Analysis tab empty | MLflow or Optuna container not running. | `make services-up` and check `docker compose ps`. |
| `database is locked` | Concurrent writes contending on SQLite. | The registry uses WAL mode; transient errors should self-recover. Persistent errors indicate stale shared-memory files (`.db-shm`, `.db-wal`); restart the GUI process. |

## Telemetry

`multiverse/gui_telemetry.py` records anonymous usage counters into a local file. Telemetry is opt-in and disabled by default. There is no remote endpoint.
