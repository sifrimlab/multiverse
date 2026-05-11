# Streamlit GUI Reference

The Multiverse GUI is a 7-tab Streamlit application that covers the full benchmarking
workflow — from dataset registration through live execution monitoring to embedded results
dashboards. Launch it with:

```bash
make setup          # installs ml-legacy deps, starts Streamlit on http://localhost:8501
```

The GUI is a **thin client**: it does not import MLflow tracking or any model framework.
It communicates with external services through HTTP (MLflow REST API, Optuna Dashboard HTTP)
and with the local registry through direct SQLite reads via `registry_db.py`.

---

## Sidebar: Observability Status

Every page load checks whether MLflow and Optuna Dashboard are reachable using a stdlib
`urllib.request` call (no subprocess). The sidebar shows:

- **Green / "MLflow"** — service is up; click "Open" to open the UI in a new tab.
- **Red / "MLflow offline"** — service is down; reminder to run `make services-up`.
- Same for Optuna.

Service URLs are read from environment variables:

| Var | Default |
|---|---|
| `MLFLOW_UI_URL` | `http://localhost:5000` |
| `MLFLOW_TRACKING_URI` | fallback for `MLFLOW_UI_URL` |
| `OPTUNA_UI_URL` | `http://localhost:8080` |
| `OPTUNA_PORT` | `8080` (used when `OPTUNA_UI_URL` is absent) |

---

## Tab 1 — 📦 Registry

Browse all registered datasets and models from the SQLite registry.

**Dataset columns:** slug, name, omics, status  
**Model columns:** slug, version, name, supported omics, status

**Register a new dataset** (expandable form):
- Toggle "Build manifest from fields" to fill a form instead of providing a YAML path.
- Fields: name, omics modalities, file paths per modality, batch key, cell type key.
- Submitting calls `multiverse.runner.cli register-dataset` and streams its output.

**Register a model** (expandable form):
- Provide a path to `model.yaml`.
- Optional "Build Docker image locally" toggle calls the CLI with `--build`.

Click **🔄 Refresh Registry** to clear the `@st.cache_data` cache after external registrations.

---

## Tab 2 — 🧬 Job Builder

Plan which dataset × model pairs to benchmark.

1. **Compatibility Matrix** — read-only colour-coded grid (green = Compatible, yellow = Partial,
   red = Incompatible) derived from each model's `supported_omics` vs the dataset's available
   omics.
2. **Select Jobs** — editable checkbox table. Incompatible rows are deselected automatically.
3. **Resource Summary** — metrics: total jobs, unique datasets, unique models, committed RAM
   (summed from each model's `memory_limit` in `model.yaml`), available host RAM.
4. **Generate Run Manifest** — writes `run_manifest.yaml` to disk; shows the YAML inline.

Selected jobs are stored in `st.session_state["planned_jobs"]` and consumed by the Parameters
and Execute tabs.

---

## Tab 3 — ⚙️ Parameters

Set per-job hyperparameter overrides before launching a run.

Each dataset × model pair gets an expander. If the model has a
`schemas/models/<slug>.hyperparameters.schema.json`, the GUI generates typed controls:

| Schema type | Widget |
|---|---|
| `enum` | Selectbox |
| `integer` | Number input (honours min/max/step) |
| `number` | Number input (honours min/max, 6 decimal places) |
| `boolean` | Checkbox |
| string / unknown | Text input |

**Sweep mode toggle:** each sweepable parameter has a "Sweep" toggle. When enabled, the
fixed widget is replaced by a range slider (integers) or two number inputs (floats) plus an
Optuna distribution selector (`uniform` / `log_uniform`). The resulting value is a search
space spec dict consumed by `tuner.py`.

Click **Generate Run Manifest (with params)** to write the manifest including all overrides.

---

## Tab 4 — 🚀 Execute

Launch a benchmark run and monitor it in real time.

### Resource Ledger

Shows three progress bars:
- **OS Used RAM** — current host usage vs total.
- **Committed Job RAM** — sum of each planned job's `memory_limit` vs the cap.
- **Free RAM (after jobs)** — remaining headroom.

"Host RAM Override" slider lets you simulate a smaller machine for wave-admission planning.

**Admission Wave Simulation table** — greedy bin-packing showing which jobs run in parallel
(Wave 1, Wave 2, …) given the RAM cap.

### Launch & Monitor

Fill in the manifest path, output directory, and random seed, then click **Launch Run**.

A `subprocess.Popen` streams the orchestrator's output line by line. The status table
updates per-job states (`⏳ Pending → 🔵 Training → 🟢 Done / 🔴 Failed`) by matching
log lines against job keys.

### Live MLflow Metrics

A `@st.fragment(run_every=timedelta(seconds=5))` panel that polls MLflow every 5 seconds
without a full page rerun. Only the metrics table re-renders.

**Requires `make services-up`** — shows an info message when MLflow is offline.

The experiment name defaults to the one from the most recent manifest (read from
`st.session_state["jb_exp_name"]`), or to the experiment linked in the Results tab. It can
be changed freely.

**Sparkline columns:** ARI, NMI, silhouette score (y-axis 0–1), loss, val_loss (auto-scaled).
Only columns that have at least one data point across all displayed runs are shown.

The underlying `fetch_live_metrics()` in `gui_utils.py` is decorated with
`@st.cache_data(ttl=5)`, giving at most one real MLflow API round-trip per 5-second window
shared across all Streamlit sessions.

---

## Tab 5 — 📊 Results

Browse runs recorded in the SQLite registry.

**Filter** by status (All / SUCCESS / FAILED / RUNNING) and **Refresh** to re-query.

**Drill-down** (SUCCESS runs only):
- Metrics table + bar chart from `metrics.json`
- Container log from `container.log` (expandable)
- Job spec from `job_spec.json` (expandable)
- Artifact directory path

**MLflow Deep-Link** (bottom of drill-down):  
The GUI tries to auto-detect the experiment name from `job_spec.json` (checks
`run_settings.experiment_name`, `globals.experiment_name`, `experiment_name`). If found,
it calls the MLflow REST API (`GET /api/2.0/mlflow/experiments/get-by-name`) to resolve the
experiment ID and stores it in `st.session_state["active_experiment_id"]`. This causes the
**🔬 Experiment Analysis** tab to open directly to that experiment.

A manual "Set experiment" expander is always shown as a fallback.

---

## Tab 6 — 🔬 Experiment Analysis

An inline MLflow UI embedded via `st.components.v1.iframe`.

- When `active_experiment_id` is set (by the Results tab drill-down), the iframe src is
  `{mlflow_base}/#/experiments/{id}` — a direct deep-link to that experiment.
- "Show all" button clears the active experiment and returns to the MLflow home page.
- A height slider (400–1200 px) lets users resize the iframe.
- A `st.link_button` above the iframe opens MLflow in a new tab — use this if the browser
  blocks the iframe due to mixed-content policy (HTTP iframe inside HTTPS page).

**Requires `make services-up`** — shows a warning and link button when MLflow is offline.

---

## Tab 7 — 📈 Sweep Tracker

An inline Optuna Dashboard embedded via `st.components.v1.iframe`.

Shows all Optuna studies in `store/optuna.db`. Use the height slider to resize.
The same mixed-content fallback (`st.link_button`) applies as in the MLflow tab.

**Requires `make services-up`** — shows a warning and link button when Optuna Dashboard is offline.

---

## Session State Keys

| Key | Type | Set by | Read by |
|---|---|---|---|
| `planned_jobs` | list[dict] | Job Builder | Parameters, Execute |
| `run_mode` | str | Job Builder | Parameters |
| `pair_params` | dict | Parameters | Job Builder (manifest gen) |
| `jb_exp_name` | str | Job Builder widget | Execute (live metrics default) |
| `active_experiment_id` | str \| None | Results drill-down | Experiment Analysis tab |
| `active_experiment_name` | str | Results drill-down | Experiment Analysis tab, Execute |
| `registry_dirty` | bool | Registry (after register) | Registry (refresh hint) |

---

## Running the GUI Inside Docker (optional)

The `streamlit` service in `docker-compose.yml` is in the `gui` Docker profile. It mounts
`/var/run/docker.sock` so the GUI can launch model containers, but it requires that Docker
socket to be accessible inside the container.

```bash
docker compose --profile gui up -d
```

For day-to-day development, `make setup` on the host is simpler and avoids Docker-in-Docker
complexities.
