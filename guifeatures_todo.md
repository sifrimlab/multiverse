# PRD — Multiverse GUI v2: MLOps Control Plane

**Status:** Design Spec  
**Audience:** Rookie bioinformatician (primary), power user (secondary)  
**Architecture constraint:** GUI is read-only on SQLite; it writes only `run_manifest.yaml` and triggers subprocesses. Zero imports from `multiverse.runner` directly.

---

## 0. Application Shell & Session State Strategy

### Tab Layout

```
st.set_page_config(layout="wide")

tabs = st.tabs([
    "📦 Registry",          # Tab 1 — Register datasets & models
    "🧬 Job Builder",       # Tab 2 — Compatibility matrix + plan
    "⚙️  Parameters",       # Tab 3 — Fixed params OR sweep config
    "🚀 Execute",           # Tab 4 — Launch + resource ledger
    "📊 Results",           # Tab 5 — Artifact browser + MLflow links
])
```

### Central `st.session_state` Schema

Streamlit reruns the entire script on every widget interaction. All mutable state lives in a single dict initialized once at the top of `main()`:

```python
_DEFAULTS = {
    # Tab 2
    "selected_datasets": [],       # list[str] — dataset names
    "selected_models":   [],       # list[str] — model names
    "planned_jobs":      [],       # list[dict] {Dataset, Model, Status}
    # Tab 3
    "run_mode":          "single", # "single" | "sweep"
    "pair_params":       {},       # (ds, model) -> dict of fixed params
    "pair_sweep":        {},       # (ds, model) -> dict of search_space specs
    # Tab 4
    "manifest_path":     None,     # str path after generation
    "host_ram_gb":       None,     # float override; None = auto-detect
    # Tab 1 side-effects
    "registry_dirty":    False,    # bust st.cache_data on next fetch
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
```

**Key rule:** Every widget that feeds the manifest uses `key=` so Streamlit owns the value across reruns. Complex objects (pair_params, pair_sweep) use explicit `st.session_state` writes via `on_change=` callbacks rather than reading widget return values, because nested data structures cannot be bound to widget keys directly.

---

## Tab 1 — Registry

### 1A. Dataset Registration Wizard

**User flow:**
1. User clicks **"Register New Dataset"** (collapsed by default; uses `st.expander`).
2. Fill a structured form — no manual YAML editing required.
3. Wizard writes a `dataset.yaml` to `store/datasets/<slug>/`, then calls `register_from_manifest()` via `subprocess.run`.
4. On success, the registry table below refreshes (`st.cache_data.clear()`).

**Streamlit components:**

```python
with st.expander("➕ Register a New Dataset", expanded=False):
    col_left, col_right = st.columns(2)
    
    with col_left:
        ds_name    = st.text_input("Dataset Name", placeholder="PBMC 10k Multiome")
        ds_slug    = st.text_input("Slug (auto-generated)", 
                                   value=slugify(ds_name), disabled=True)
        ds_path    = st.text_input("Path to .h5mu / .h5ad file")
        # st.file_uploader is NOT used — files are already on the host filesystem
    
    with col_right:
        omics       = st.multiselect("Omics Available", ["rna", "atac", "adt", "prot"])
        batch_key   = st.text_input("Batch Key (obs column)", placeholder="batch")
        ct_key      = st.text_input("Cell Type Key (obs column)", placeholder="cell_type")
    
    if st.button("Register Dataset", type="primary"):
        # 1. Write dataset.yaml
        # 2. subprocess.run(["python", "-m", "multiverse.runner.cli", "register-dataset", ...])
        # 3. Show result in st.toast()
        st.session_state["registry_dirty"] = True
```

**Inline path validation:** Before the button becomes active, show a green ✅ or red ❌ next to the path field using `os.path.exists()` — no round-trip needed.

```python
if ds_path:
    exists = os.path.exists(ds_path)
    st.caption("✅ File found" if exists else "❌ File not found")
    register_btn_disabled = not exists
```

**Backend connection:** Calls `register_from_manifest()` from `multiverse/ingestion.py` via subprocess. Reads from `datasets` table for the table below.

---

### 1B. Model Registration + Local Build Wizard

**User flow:**
1. User selects a model YAML from a path input or from `store/models/`.
2. GUI previews the parsed fields (name, version, image, supported omics).
3. Optional toggle: **"Build Docker image locally"** — triggers the builder with live log streaming.
4. On completion, the models table refreshes.

**Streamlit components:**

```python
with st.expander("➕ Register / Build a Model", expanded=False):
    manifest_path = st.text_input("Path to model.yaml")
    
    col1, col2 = st.columns([3,1])
    with col1:
        build_locally = st.toggle("Build Docker image locally after registration", value=False)
        # Tooltip: "Enable if the model is private or not on Docker Hub"
    with col2:
        st.caption("Requires Docker daemon running on this host.")
    
    if manifest_path and os.path.exists(manifest_path):
        # Preview parsed manifest — no subprocess needed, just yaml.safe_load
        with open(manifest_path) as f:
            preview = yaml.safe_load(f)
        st.json(preview, expanded=False)
    
    if st.button("Register Model", type="primary"):
        # Streaming build log using st.status
        with st.status("Registering model...", expanded=True) as build_status:
            proc = subprocess.Popen(
                ["python", "-m", "multiverse.runner.cli", 
                 "register-model", "--manifest", manifest_path,
                 *(["--build"] if build_locally else [])],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            log_area = st.empty()
            log_lines = []
            for line in proc.stdout:
                log_lines.append(line)
                log_area.code("".join(log_lines[-40:]))  # rolling 40-line window
            proc.wait()
            if proc.returncode == 0:
                build_status.update(label="✅ Model registered!", state="complete")
            else:
                build_status.update(label="❌ Build failed", state="error")
        st.session_state["registry_dirty"] = True
```

**Why `st.status` + `subprocess.Popen`:** `st.status` (added Streamlit 1.28) renders an expandable status card with states `running` / `complete` / `error`. The `Popen` read loop gives us line-by-line streaming without blocking the event loop. This is the canonical pattern for long-running subprocesses in Streamlit.

**Backend connection:** `subprocess.Popen` → CLI → `register_model_from_manifest()` + `build_local_model()` from `multiverse/builder.py`. Reads from `models` table for the table below.

---

### 1C. Registry Health Table

Below both wizards, show two `st.dataframe` tables — one for datasets, one for models. These are styled with `st.dataframe(df.style.apply(...))` using color-coded `status` column:

```python
@st.cache_data(ttl=30)    # Auto-refresh every 30s; manual bust via registry_dirty flag
def fetch_registry():
    if st.session_state.get("registry_dirty"):
        st.cache_data.clear()
        st.session_state["registry_dirty"] = False
    datasets = get_all_datasets()   # registry_db.py
    models   = get_all_models()     # registry_db.py — already returns latest ACTIVE per slug
    return datasets, models
```

Color coding: `READY`/`ACTIVE` → green, `FAILED` → red, `PENDING` → yellow.

---

## Tab 2 — Job Builder & Compatibility Matrix

### 2A. The Compatibility Matrix

**The core UX insight:** The existing `gui.py` renders the matrix as a static `st.dataframe`. The upgrade makes it *interactive* — clicking a cell selects the job.

**Approach — `st.data_editor` with checkbox injection:**

```python
matrix_df = generate_compatibility_matrix(datasets, models)
# matrix_df: rows=datasets, cols=models, values="Compatible"/"Partial"/"Incompatible"

# Convert to a boolean selection matrix pre-populated with Compatible cells
selection_df = matrix_df.apply(
    lambda col: col.map({"Compatible": True, "Partial": False, "Incompatible": False})
)

edited = st.data_editor(
    selection_df,
    use_container_width=True,
    column_config={
        col: st.column_config.CheckboxColumn(
            col, 
            help=f"Run {col} on this dataset",
            default=False
        )
        for col in selection_df.columns
    },
    disabled=False,   # user can toggle individual cells
)
```

**Compatibility overlay:** Render a second `st.dataframe` beneath the editor showing the color-coded compatibility reason, so users understand *why* a cell is grey:

```python
st.caption("Compatibility legend: 🟢 Compatible  🟡 Partial (missing optional omics)  🔴 Incompatible")
```

**Incompatible cell guard:** Cells where the underlying `matrix_df` value is `"Incompatible"` should be disabled. This is done by overriding the `disabled` parameter per-column if the entire column is incompatible, or with a post-edit validation step:

```python
# Post-edit: revert any incompatible selections
for col in edited.columns:
    mask = matrix_df[col] == "Incompatible"
    edited.loc[mask, col] = False
```

**Deriving planned_jobs from the editor:**

```python
planned_jobs = []
for ds in edited.index:
    for model in edited.columns:
        if edited.loc[ds, model]:
            planned_jobs.append({
                "Dataset": ds,
                "Model": model,
                "Status": matrix_df.loc[ds, model],
            })
st.session_state["planned_jobs"] = planned_jobs
```

**Backend connection:** `generate_compatibility_matrix()` from `multiverse/registry.py`. Reads `datasets` and `models` tables.

---

### 2B. Dual-Axis Mode Selector

Above the matrix, a `st.radio` in a `st.columns` layout:

```python
col_axis, col_info = st.columns([2, 3])
with col_axis:
    axis = st.radio(
        "Benchmark axis",
        options=["Data-Centric (1 dataset × N models)", 
                 "Model-Centric (1 model × N datasets)",
                 "Full Grid (N × M)"],
        horizontal=True,
    )
```

When axis changes, the matrix filtering logic adjusts: Data-Centric disables all columns after the first selected dataset's row. This is enforced in the `disabled` parameter of `st.data_editor`.

---

### 2C. Job Plan Summary

After the matrix editor, a compact `st.metric` row shows the plan at a glance:

```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jobs Planned", len(planned_jobs))
col2.metric("Datasets", len({j["Dataset"] for j in planned_jobs}))
col3.metric("Models",   len({j["Model"]   for j in planned_jobs}))
col4.metric("Est. RAM",  f"{sum_mem:.0f} GiB",
            delta=f"{available_ram:.0f} GiB available",
            delta_color="normal" if sum_mem <= available_ram else "inverse")
```

The RAM estimate uses `mem_limit` from each model's manifest (default `16g`), parsed with the existing `_parse_mem_gb()`. This is the first touch of the Resource Ledger in the UI.

---

## Tab 3 — Parameters & Sweep Configurator

### 3A. Single Run vs. Sweep Toggle

```python
run_mode = st.segmented_control(     # Streamlit 1.40+
    "Execution Mode",
    options=["Single Run", "Optuna Sweep"],
    default="Single Run",
    key="run_mode_widget",
)
# st.segmented_control renders as a pill toggle — cleaner than radio for 2 options.
# Fall back to st.radio for older Streamlit versions.
```

The mode is stored in `st.session_state["run_mode"]`. **Switching mode does not clear existing params** — the same `pair_params` dict is kept; sweep mode adds a parallel `pair_sweep` dict.

---

### 3B. Per-Job Parameter Panels (Single Run Mode)

For each `planned_job` in `st.session_state["planned_jobs"]`, render one `st.expander`. The existing `_render_param_field()` from `gui.py` is reused unchanged — this is good separation.

**New addition: schema completeness indicator**

```python
with st.expander(f"{ds} × {model}", expanded=False):
    schema = _load_hyperparameter_schema(model_name_to_schema_path[model])
    if schema:
        n_props = len(schema.get("properties", {}))
        st.caption(f"{n_props} tunable parameters | Schema version: {schema.get('$id', 'unknown')}")
        # render fields via _render_param_field() ...
    else:
        st.warning("No JSON Schema found. Using raw JSON override.")
        # text_area fallback (already in gui.py)
```

**Session state binding via on_change:**

Each numeric/select widget uses `key=f"param::{ds}::{model}::{param_name}"`. After all expanders render, a reconciliation step reads back current widget values:

```python
def _collect_params_from_widgets(planned_jobs, schema_map):
    result = {}
    for job in planned_jobs:
        ds, mod = job["Dataset"], job["Model"]
        schema = _load_hyperparameter_schema(schema_map.get(mod))
        if not schema:
            continue
        collected = {}
        for pname in schema.get("properties", {}):
            key = f"param::{ds}::{mod}::{pname}"
            if key in st.session_state:
                collected[pname] = st.session_state[key]
        result[(ds, mod)] = collected
    return result
```

This call runs at the bottom of Tab 3, updating `st.session_state["pair_params"]`.

---

### 3C. Sweep Configurator (Optuna Mode)

When `run_mode == "Optuna Sweep"`, each param field is replaced by a **distribution editor** derived from the same JSON schema. The schema property type maps to the Optuna distribution type:

| JSON Schema type | Optuna distribution | GUI widgets |
|---|---|---|
| `integer` with min/max | `int` | `st.number_input` × 2 (low, high) + `st.number_input` step |
| `number` with min/max | `loguniform` (default) or `uniform` | `st.number_input` × 2 + `st.radio(Log scale?)` |
| `string` with `enum` | `categorical` | `st.multiselect` (choose which values to try) |
| `boolean` | `categorical` | Fixed `[True, False]`, no widget needed |

```python
def _render_sweep_field(job_key: str, param_name: str, spec: dict):
    ptype = spec.get("type")
    key_prefix = f"sweep::{job_key}::{param_name}"
    
    if spec.get("enum"):
        choices = st.multiselect(
            f"{param_name} — categorical choices",
            options=spec["enum"],
            default=spec["enum"],
            key=f"{key_prefix}::choices",
        )
        return {"type": "categorical", "choices": choices}
    
    if ptype == "integer":
        c1, c2, c3 = st.columns(3)
        low  = c1.number_input("low",  value=int(spec.get("minimum", 1)),  key=f"{key_prefix}::low")
        high = c2.number_input("high", value=int(spec.get("maximum", 100)), key=f"{key_prefix}::high")
        step = c3.number_input("step", value=1, min_value=1,               key=f"{key_prefix}::step")
        return {"type": "int", "low": low, "high": high, "step": step}
    
    if ptype == "number":
        c1, c2, c3 = st.columns(3)
        low     = c1.number_input("low",    value=float(spec.get("minimum", 1e-5)), format="%.2e", key=f"{key_prefix}::low")
        high    = c2.number_input("high",   value=float(spec.get("maximum", 1.0)),  format="%.2e", key=f"{key_prefix}::high")
        log_scale = c3.checkbox("Log scale", value=True,                           key=f"{key_prefix}::log")
        dist_type = "loguniform" if log_scale else "uniform"
        return {"type": dist_type, "low": low, "high": high}
    
    return None  # unsupported type — skip
```

**Sweep global settings** (rendered at the top of the sweep panel):

```python
st.markdown("#### Sweep Settings (apply to all jobs)")
col1, col2, col3 = st.columns(3)
n_trials       = col1.number_input("Trials per job", min_value=5, value=20, step=5)
optimize_metric = col2.text_input("Optimize metric", value="silhouette_score",
                                  help="Dot-path into metrics.json, e.g. 'batch.ilisi'")
direction      = col3.radio("Direction", ["maximize", "minimize"], horizontal=True)
study_storage  = st.text_input("Optuna storage URI", value="sqlite:///optuna.db",
                               help="File-based for local; can be postgres:// for shared")
```

These are written into `globals` in the final manifest via `build_run_manifest()`.

---

## Tab 4 — Execute

### 4A. Resource Ledger Visualizer

This is the most novel component. It reads `psutil` to show current host state, then simulates the admission queue given the planned jobs.

```python
import psutil

total_gb    = psutil.virtual_memory().total / (1024**3)
avail_gb    = psutil.virtual_memory().available / (1024**3)
used_gb     = total_gb - avail_gb

# Override for testing/staging
host_ram_gb_override = st.number_input(
    "Host RAM override (GiB, 0 = auto-detect)",
    value=0.0, step=4.0, format="%.0f",
    help="Set to simulate a smaller host for testing the scheduler. 0 = use real psutil value."
)
effective_total = host_ram_gb_override if host_ram_gb_override > 0 else total_gb
st.session_state["host_ram_gb"] = effective_total if host_ram_gb_override > 0 else None
```

**Gauge visualization using `st.progress`:**

```python
st.markdown("##### Host Memory")
col_gauge, col_stats = st.columns([3, 1])

with col_gauge:
    committed_gb = sum(_parse_mem_gb(j.get("mem_limit", "16g")) for j in planned_jobs)
    # Segment bar: [used (OS) | committed (jobs) | free]
    used_fraction      = min(used_gb / effective_total, 1.0)
    committed_fraction = min(committed_gb / effective_total, 1.0)
    
    st.progress(used_fraction, text=f"OS in use: {used_gb:.1f} GiB")
    st.progress(min(used_fraction + committed_fraction, 1.0),
                text=f"+ Jobs committed: {committed_gb:.1f} GiB")

with col_stats:
    st.metric("Total RAM",    f"{effective_total:.0f} GiB")
    st.metric("Free",         f"{(effective_total - used_gb - committed_gb):.1f} GiB",
              delta_color="off")
    st.metric("Jobs queued",  len(planned_jobs))
```

**Admission simulation table:** Show users which jobs will run concurrently vs. queue:

```python
def simulate_admission_order(jobs: list[dict], total_gb: float) -> list[dict]:
    available = total_gb
    wave, result = 1, []
    for job in jobs:
        gb = _parse_mem_gb(job.get("mem_limit", "16g"))
        if gb > total_gb:
            result.append({**job, "wave": "BLOCKED", "reason": "Exceeds total RAM"})
        elif gb <= available:
            available -= gb
            result.append({**job, "wave": wave, "reason": f"Admits ({gb:.0f} GiB)"})
        else:
            wave += 1
            available = total_gb - gb
            result.append({**job, "wave": wave, "reason": f"Waits, then admits ({gb:.0f} GiB)"})
    return result

if planned_jobs:
    sim = simulate_admission_order(planned_jobs, effective_total)
    sim_df = pd.DataFrame(sim)[["Dataset", "Model", "mem_limit", "wave", "reason"]]
    st.dataframe(
        sim_df.style.apply(
            lambda row: ["background-color: #d4edda" if row["wave"] == 1
                         else "background-color: #fff3cd" if isinstance(row["wave"], int)
                         else "background-color: #f8d7da"] * len(row),
            axis=1,
        ),
        use_container_width=True,
    )
```

**Backend connection:** Pure Python — reads `planned_jobs` from session state, calls `_parse_mem_gb()` (importable since it has no Docker dependency), reads psutil.

---

### 4B. Manifest Generation + Launch

```python
st.divider()
st.markdown("#### Generate Manifest & Run")

exp_name = st.text_input("Experiment Name", value="benchmark_run", key="exp_name")
seed     = st.number_input("Random Seed", value=42, key="seed_global")

col_gen, col_run = st.columns(2)

with col_gen:
    if st.button("💾 Save Manifest", use_container_width=True):
        manifest = build_run_manifest(
            experiment_name=exp_name,
            random_seed=seed,
            run_mode=st.session_state["run_mode"],
            planned_jobs=st.session_state["planned_jobs"],
            dataset_name_to_slug=dataset_name_to_slug,
            pair_params=st.session_state["pair_params"],
            # NEW: inject host_ram_gb and mem_limit per job
        )
        if st.session_state["host_ram_gb"]:
            manifest["globals"]["host_ram_gb"] = st.session_state["host_ram_gb"]
        
        path = "run_manifest.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)
        st.session_state["manifest_path"] = path
        st.success(f"Saved to `{path}`")
        st.code(yaml.safe_dump(manifest), language="yaml")

with col_run:
    run_disabled = st.session_state.get("manifest_path") is None
    if st.button("🚀 Launch Run", type="primary", 
                 use_container_width=True, disabled=run_disabled):
        st.session_state["run_proc_started"] = True
```

---

### 4C. Real-Time DAG Progress (Execution Monitor)

When `run_proc_started` is True, the GUI launches the orchestrator in a subprocess and reads its stdout line-by-line inside `st.status`.

**The DAG stages map to status strings emitted by `run_workflow_async`:**

| Log pattern | GUI Stage | Icon |
|---|---|---|
| `Starting container for` | Training | 🔵 |
| `ResourcePool ... Admitted` | Admitted | 🟡 |
| `Released ... GiB` | Released | ⬜ |
| `completed successfully` | Promoted | 🟢 |
| `failed with exit code` | Failed | 🔴 |
| `INSUFFICIENT_RESOURCES` | Blocked | ⛔ |

```python
if st.session_state.get("run_proc_started"):
    proc = subprocess.Popen(
        ["python", "-m", "multiverse.runner.cli", "run",
         "--manifest", st.session_state["manifest_path"],
         "--output", f"store/artifacts/{slugify(exp_name)}",
         "--seed",   str(seed)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    
    job_statuses = {j["Dataset"] + "_" + j["Model"]: "⏳ Queued"
                   for j in st.session_state["planned_jobs"]}
    
    with st.status("Running experiment...", expanded=True) as run_status:
        status_placeholder = st.empty()
        
        for line in proc.stdout:
            # Parse log line to update job_statuses dict
            for job_name in job_statuses:
                if job_name in line:
                    if "Admitted"              in line: job_statuses[job_name] = "🟡 Admitted"
                    elif "Starting container"  in line: job_statuses[job_name] = "🔵 Training"
                    elif "completed"           in line: job_statuses[job_name] = "🟢 Done"
                    elif "failed"              in line: job_statuses[job_name] = "🔴 Failed"
                    elif "INSUFFICIENT"        in line: job_statuses[job_name] = "⛔ Blocked"
            
            # Render live table
            status_df = pd.DataFrame(
                [{"Job": k, "Status": v} for k, v in job_statuses.items()]
            )
            status_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)
        
        proc.wait()
        if proc.returncode == 0:
            run_status.update(label="✅ Experiment complete!", state="complete")
            st.session_state["registry_dirty"] = True  # bust cache for Tab 5
        else:
            run_status.update(label="❌ Experiment failed — check logs", state="error")
    
    st.session_state["run_proc_started"] = False
```

**Why not `asyncio.run()` inside Streamlit:** Streamlit runs inside its own event loop. Calling `asyncio.run()` raises `RuntimeError: This event loop is already running`. The subprocess approach is the correct boundary — the GUI is a control plane, the orchestrator owns async execution.

---

## Tab 5 — Results & Artifact Browser

### 5A. Run History Table

```python
@st.cache_data(ttl=15)
def fetch_runs():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    runs = conn.execute("""
        SELECT r.run_id, d.name as dataset, r.model_name, r.model_version,
               r.status, r.output_path
        FROM runs r
        JOIN datasets d ON d.id = r.dataset_id
        ORDER BY r.run_id DESC
        LIMIT 200
    """).fetchall()
    conn.close()
    return [dict(r) for r in runs]

runs = fetch_runs()
runs_df = pd.DataFrame(runs)

# Color-code status column
status_colors = {
    "SUCCESS": "background-color: #d4edda",
    "FAILED":  "background-color: #f8d7da",
    "FAILED: INSUFFICIENT_RESOURCES": "background-color: #fff3cd",
}
```

**Filter row above the table:**

```python
col_f1, col_f2, col_f3 = st.columns(3)
filter_ds    = col_f1.multiselect("Dataset",  runs_df["dataset"].unique())
filter_model = col_f2.multiselect("Model",    runs_df["model_name"].unique())
filter_status= col_f3.multiselect("Status",   runs_df["status"].unique())

mask = pd.Series([True] * len(runs_df))
if filter_ds:     mask &= runs_df["dataset"].isin(filter_ds)
if filter_model:  mask &= runs_df["model_name"].isin(filter_model)
if filter_status: mask &= runs_df["status"].isin(filter_status)
```

**Backend connection:** Direct read from `runs` table via `get_db_connection()`. Never imports docker or runner modules.

---

### 5B. Artifact Drill-Down

```python
selected_run = st.selectbox(
    "Drill into run",
    options=runs_df[runs_df["status"] == "SUCCESS"]["run_id"].tolist(),
    format_func=lambda rid: f"Run #{rid} — {runs_df.loc[runs_df.run_id==rid, 'model_name'].iloc[0]}",
)

if selected_run:
    output_path = runs_df.loc[runs_df.run_id == selected_run, "output_path"].iloc[0]
    metrics_path = os.path.join(output_path, "metrics.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Flatten nested metrics for display
        flat = _to_flat_float_metrics(metrics)   # reuse from tracking.py logic
        metrics_df = pd.DataFrame(flat.items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Bar chart of key bio metrics
        bio_keys = [k for k in flat if any(x in k for x in ["silhouette","ilisi","clisi","nmi","ari"])]
        if bio_keys:
            st.bar_chart({k: flat[k] for k in bio_keys})
    
    # Log viewer
    log_path = os.path.join(output_path, "container.log")
    if os.path.exists(log_path):
        with st.expander("Container Log", expanded=False):
            with open(log_path) as f:
                st.code(f.read()[-5000:], language="text")  # last 5KB
    
    # MLflow link (if tracking URI is file-based, link won't work remotely — show path only)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    st.caption(f"MLflow tracking: `{mlflow_uri}` — run `mlflow ui` to browse.")
```

---

## Cross-Cutting Concerns

### Error Surface Design

Errors are never silent. Three levels:
1. `st.toast("...", icon="⚠️")` — transient warnings (e.g., schema missing)
2. `st.warning(...)` — persistent inline (e.g., no datasets registered)
3. `st.error(...)` — blocking (e.g., manifest parse failure, incompatible selection)

The `FAILED: INSUFFICIENT_RESOURCES` status from the Resource Ledger must be called out explicitly:

```python
if any("INSUFFICIENT_RESOURCES" in j.get("status", "") for j in runs):
    st.warning(
        "One or more jobs were blocked by the memory ledger. "
        "Increase `host_ram_gb` or reduce job `mem_limit` values.",
        icon="⚠️"
    )
```

### Rookie vs. Power User Progressive Disclosure

| Feature | Rookie sees | Power user unlocks |
|---|---|---|
| Param config | Expander collapsed, defaults pre-filled | Expand, see all JSON schema fields |
| Sweep | Hidden behind `st.segmented_control` toggle | Full distribution editor per param |
| Resource override | Not visible | `st.expander("Advanced: Resource Ledger Override")` |
| Manifest YAML | Shown as preview only | `st.download_button` to save and edit manually |
| CLI equivalent | Not shown | `st.code("make benchmark config=run_manifest.yaml")` |

### `st.cache_data` Invalidation Contract

| Cache | TTL | Manual bust trigger |
|---|---|---|
| `fetch_registry()` | 30s | `st.session_state["registry_dirty"] = True` |
| `fetch_runs()` | 15s | After any subprocess returns `returncode == 0` |
| `generate_compatibility_matrix()` | Session (no TTL) | On `registry_dirty` |

---

## What Stays Unchanged from `gui.py`

- `build_run_manifest()` — pure function, no changes needed
- `_render_param_field()` — reused in Tab 3 Single Run mode
- `_load_hyperparameter_schema()` — reused in both Tab 3 modes
- `slugify_experiment_name()` — reused in Tab 4
- `fetch_registry_data()` — renamed `fetch_registry()` with TTL

The new GUI is an additive expansion. No existing logic is deleted — it's promoted into the appropriate tab.