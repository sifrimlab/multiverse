# Observability

mvexp ships two observability services as part of `docker-compose.yml`: an MLflow tracking server and an Optuna Dashboard. They are launched by `make services-up` and stopped by `make services-down`. The Streamlit **Analysis** tab embeds both, but each is also a fully featured standalone web UI.

This page documents what gets logged where, how the components are wired, and how to verify connectivity. It is written for platform operators; researchers can usually rely on the embedded views without thinking about the plumbing.

## Service Map

| Service | URL | Container | Backing store | Purpose |
|---|---|---|---|---|
| MLflow | `http://localhost:5000` | `mvr-mlflow` | `store/mlflow.db` (SQLite) + `store/artifacts/` | Cross-run parameter and metric comparison, artifact browser. |
| Optuna Dashboard | `http://localhost:8080` | `mvr-optuna` | `store/optuna.db` (SQLite) | Sweep visualization, parameter importance, pruning history. |
| Streamlit (optional, profile `gui`) | `http://localhost:8501` | `mvr-streamlit` | host bind-mount + Docker socket | Runs the GUI inside a container for shared lab installs. |

Ports are configurable via `MLFLOW_PORT`, `OPTUNA_PORT`, and `STREAMLIT_PORT`. All three services bind-mount `./store` to `/data`, so they read and write the same SQLite databases as the host orchestrator.

## MLflow Run Model

Each containerized model execution corresponds to **one MLflow run**, opened by the host orchestrator *before* the container launches and closed *after* it exits. The run captures four kinds of data:

1. **Hyperparameters and tags** — logged at run start by `multiverse.tracking.start_parent_mlflow_run()`, so they are visible in MLflow before training begins.
2. **System metrics** — CPU, GPU, and RAM utilization sampled by MLflow's built-in monitor while the parent run is open (i.e., the duration of the container).
3. **Per-epoch metrics** — streamed from inside the container by `mvr_worker.EpochLogger`. The host injects `MLFLOW_RUN_ID` into the container environment, and `EpochLogger` attaches to that run instead of opening a duplicate. Models without per-epoch hooks (e.g. PCA) skip this step.
4. **Final scalars and artifacts** — appended by the host after the container exits via `log_successful_run_to_mlflow()`, which sanitises `NaN`/`±Inf`, flattens nested metric dictionaries, and then closes the run with status `FINISHED` or `FAILED`.

Optuna sweeps appear as **child runs** under a parent MLflow run that represents the study, so a sweep's trials remain navigable as a group.

### Network Configuration

The Docker runner forwards `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` into each model container. On Linux, `localhost` / `127.0.0.1` URIs are rewritten to `host.docker.internal` and the container is started with `--add-host=host.docker.internal:host-gateway`, so containers can reach an MLflow server bound on the host loopback.

If the MLflow service is itself running inside Docker (the default after `make services-up`), the URI resolves to `http://mlflow:5000` over the compose network.

If your MLflow server is bound to `127.0.0.1` only on the host, start it with `--host 0.0.0.0` so the gateway can route container traffic.

## Optuna Study Model

When a manifest specifies `globals.run_gridsearch: true`, `multiverse/runner/tuner.py` creates one Optuna study per job. Studies are persisted to `store/optuna.db` and surfaced in the Optuna Dashboard.

| Concept | Where it lives |
|---|---|
| Study name | `<experiment_name>__<dataset_slug>__<model_slug>` |
| Trial parameters | Sampled from the model's hyperparameter schema; sweepable fields are flagged `x-sweepable: true`. |
| Trial metric | Configured via `globals.metrics`; defaults to a primary bio-conservation metric. |
| Pruning | MedianPruner by default; configurable per-job in future releases. |

Each trial also logs to MLflow as a child of the study's parent run, so the same numerical comparison is available in either UI.

## Local Sidecars

Even with both services running, each successful run writes two local sidecars to its artifact directory:

| File | Contents |
|---|---|
| `metrics.json` | Final scalars and (optionally) a `history` block. |
| `metrics.jsonl` | One JSON object per epoch (`step`, `timestamp`, metrics) emitted by `EpochLogger`. Survives crashes. |

The artifact tree is therefore self-contained: even if `mlflow.db` is lost, the per-run record is recoverable from disk.

## Verifying the Wiring

```bash
make services-up
make status                                # docker compose ps with port bindings
curl -sf http://localhost:5000/health      # MLflow health endpoint
curl -sf http://localhost:8080/            # Optuna Dashboard root
```

Inside a model container, the SDK's `EpochLogger` will log a warning and fall back to JSONL-only mode if `MLFLOW_TRACKING_URI` is unset or unreachable. A run is never *failed* solely because tracking is unavailable.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `Analysis` tab is blank | Service not running. | `make services-up`; check `make status`. |
| MLflow has no entry for a successful run | Container could not reach the tracking server. | Confirm `MLFLOW_TRACKING_URI` resolves from inside the container. |
| Duplicate runs in MLflow | A model container opened its own run instead of attaching to `MLFLOW_RUN_ID`. | Use `EpochLogger` from `mvr-worker`; do not call `mlflow.start_run()` directly in container code. |
| Optuna Dashboard empty | No study has been created yet. | Run a manifest with `run_gridsearch: true`. |
| `store/mlflow.db-wal` growing large | Many concurrent writes. | Expected during heavy benchmarking; SQLite checkpoints periodically. |
