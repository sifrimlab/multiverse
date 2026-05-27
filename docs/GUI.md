# Streamlit GUI

The Streamlit app at `multiverse/gui.py` is the primary researcher interface. It exposes registration, benchmark planning, mvd-backed execution, results browsing, and analysis dashboards.

## Layout

| Tab | Query parameter | Purpose |
|---|---|---|
| Registry | `?tab=registry` | Inspect, register, and refresh datasets and models. |
| Configure | `?tab=configure` | Pick dataset x model pairs and set hyperparameters. |
| Run | `?tab=run` | Submit and monitor benchmark execution. |
| Results | `?tab=results` | Browse completed runs, metrics, logs, and artifacts. |
| Analysis | `?tab=analysis` | Embedded MLflow and Optuna views for cross-run analysis. |

## Registry

The Registry tab creates and refreshes dataset/model rows in the local SQLite index. Registration paths are hardened: path escapes are rejected, and elevated Docker options in model manifests require explicit opt-in.

## Configure

Configure builds `run_manifest.yaml` from selected compatible dataset/model pairs. The compatibility matrix is computed from registered dataset omics and model requirements. Hyperparameter forms are rendered from model JSON schemas.

## Run

The Run tab uses the in-process mvd controller. It does not spawn `multiverse.runner.cli` as a subprocess and does not own Docker containers directly.

- **Launch Run** parses the manifest, submits jobs through the kernel/client boundary, and records mvd attempt IDs.
- **Cancel Run** calls the kernel cancellation verb; it does not kill a local process handle.
- The status table renders kernel states such as `RUNNING`, `PROMOTING`, `ARTIFACT_SUCCESS`, `FAILED`, and `CANCELLED`.
- The event panel shows state transitions observed from kernel queries. Container logs remain artifact files after the run produces or preserves a workspace.

## Results

Results reads from SQLite for fast listing and from artifact directories for durable run evidence. A successful run is trustworthy only when its artifact manifest and checksum sidecar verify.

## Analysis

MLflow and Optuna are comparison/projection surfaces. They are not the source of scientific truth. If MLflow is offline, a run can still reach `ARTIFACT_SUCCESS`; sync can be retried later with `multiverse mlflow-sync`.

## Common Issues

| Symptom | Likely cause | What to do |
|---|---|---|
| Dataset missing from Configure | Registry cache stale. | Open Registry and refresh. |
| Launch fails before container start | Manifest references stale rows or Docker is unavailable. | Regenerate the manifest or run `multiverse doctor`. |
| Run is `FAILED` | Container exit, validation refusal, or Docker error. | Inspect the event panel, `failure_reason`, and preserved logs. |
| MLflow panel is empty | Projection service is offline. | Start services or sync later; artifact bundles remain authoritative. |
| SQLite listing looks stale | Index drift. | Run `multiverse rebuild-index`. |

## Telemetry

`multiverse/gui_telemetry.py` records anonymous usage counters into a local file. Telemetry is opt-in and disabled by default. There is no remote endpoint.
