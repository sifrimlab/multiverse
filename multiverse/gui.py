import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # type: ignore[import-untyped]
import yaml
from multiverse.gui_utils import LIVE_METRIC_KEYS, fetch_live_metrics, render_hyperparameters_form
from multiverse.registry import generate_compatibility_matrix
from multiverse.registry_db import get_all_datasets, get_all_models, get_db_connection, init_db


# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict = {
    "selected_datasets": [],
    "selected_models": [],
    "planned_jobs": [],
    "run_mode": "Use User Params",
    "registry_dirty": False,
    "active_experiment_id": None,
    "active_experiment_name": "",
}


def _init_session_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data
def fetch_registry_data():
    init_db()
    datasets = get_all_datasets()
    models = get_all_models()
    return datasets, models


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def slugify_experiment_name(raw_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_name.strip()).strip("-").lower()
    if not slug:
        raise ValueError("Experiment Name must contain at least one alphanumeric character.")
    return slug


def build_run_manifest(
    *,
    experiment_name: str,
    random_seed: int,
    run_mode: str,
    planned_jobs: list[dict],
    dataset_name_to_slug: dict[str, str],
    pair_params: dict[tuple[str, str], dict],
) -> dict:
    run_user_params = run_mode == "Use User Params"
    run_gridsearch = run_mode == "Run Gridsearch"
    manifest = {
        "globals": {
            "experiment_name": slugify_experiment_name(experiment_name),
            "random_seed": int(random_seed),
            "run_user_params": run_user_params,
            "run_gridsearch": run_gridsearch,
        },
        "jobs": [],
    }
    for job in planned_jobs:
        ds_name = job["Dataset"]
        mod_name = job["Model"]
        manifest["jobs"].append(
            {
                "dataset_slug": dataset_name_to_slug[ds_name],
                "model_name": mod_name,
                "model_params": pair_params.get((ds_name, mod_name), {}) or {},
            }
        )
    return manifest


def _parse_memory_gb(mem_str: str) -> float:
    """Parse Docker-style memory strings ('16g', '4096m', '2t') to GiB."""
    if not mem_str:
        return 0.0
    s = mem_str.strip().lower()
    try:
        if s.endswith("t"):
            return float(s[:-1]) * 1024
        if s.endswith("g"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) / 1024
        if s.endswith("k"):
            return float(s[:-1]) / (1024 * 1024)
        return float(s) / (1024 ** 3)
    except ValueError:
        return 0.0


def _load_hyperparameter_schema(schema_path: str | None) -> dict | None:
    if not schema_path:
        return None
    path = Path(schema_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        return schema if isinstance(schema, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Subprocess streaming helper
# ---------------------------------------------------------------------------

def _stream_subprocess(cmd: list[str], status_label: str) -> bool:
    with st.status(status_label, expanded=True) as status:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            st.write(line.rstrip())
        proc.wait()
        if proc.returncode == 0:
            status.update(label=f"{status_label} — Done ✓", state="complete")
            return True
        else:
            status.update(label=f"{status_label} — Failed ✗", state="error")
            return False


# ---------------------------------------------------------------------------
# Tab: Registry
# ---------------------------------------------------------------------------

def _render_registry_tab() -> None:
    st.header("Asset Registry")

    if st.button("🔄 Refresh Registry"):
        fetch_registry_data.clear()
        st.session_state["registry_dirty"] = False
        st.rerun()

    datasets, models = fetch_registry_data()

    if not datasets and not models:
        st.info("No assets found. Register a dataset or model below.")
    else:
        col_ds, col_md = st.columns(2)

        with col_ds:
            st.subheader("Datasets")
            if datasets:
                ds_rows = [
                    {
                        "slug": d.get("slug", ""),
                        "name": d.get("name", ""),
                        "omics": d.get("omics_available", ""),
                        "status": d.get("status", ""),
                    }
                    for d in datasets
                ]
                st.dataframe(pd.DataFrame(ds_rows), width='stretch')
            else:
                st.caption("No datasets registered.")

        with col_md:
            st.subheader("Models")
            if models:
                md_rows = [
                    {
                        "slug": m.get("slug", ""),
                        "version": m.get("version", ""),
                        "name": m.get("name", ""),
                        "omics": m.get("supported_omics", ""),
                        "status": m.get("status", ""),
                    }
                    for m in models
                ]
                st.dataframe(pd.DataFrame(md_rows), width='stretch')
            else:
                st.caption("No models registered.")

    st.divider()

    # --- Register dataset ---
    with st.expander("➕ Register New Dataset", expanded=False):
        use_fields = st.toggle(
            "Build manifest from fields (I don't have a dataset.yaml yet)",
            key="ds_use_fields",
        )

        if use_fields:
            ds_name = st.text_input("Dataset name", key="ds_name")
            ds_omics = st.multiselect(
                "Available omics",
                options=["rna", "atac", "adt", "other"],
                key="ds_omics",
            )
            ds_rna_path = st.text_input(
                "Path to RNA .h5ad (leave blank if not applicable)", key="ds_rna_path"
            )
            ds_atac_path = st.text_input(
                "Path to ATAC .h5ad (leave blank if not applicable)", key="ds_atac_path"
            )
            ds_adt_path = st.text_input(
                "Path to ADT .h5ad (leave blank if not applicable)", key="ds_adt_path"
            )
            ds_batch_key = st.text_input("batch_key (optional)", key="ds_batch_key")
            ds_cell_type_key = st.text_input("cell_type_key (optional)", key="ds_cell_type_key")

            if st.button("Register Dataset", key="btn_register_ds_fields"):
                if not ds_name.strip():
                    st.error("Dataset name is required.")
                elif not ds_omics:
                    st.error("Select at least one omics modality.")
                else:
                    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", ds_name.strip()).strip("-").lower()
                    manifest_dir = Path("store/datasets") / slug
                    manifest_dir.mkdir(parents=True, exist_ok=True)
                    manifest_path = manifest_dir / "dataset.yaml"

                    raw_files: dict[str, str] = {}
                    for modality, field_val in [
                        ("rna", ds_rna_path),
                        ("atac", ds_atac_path),
                        ("adt", ds_adt_path),
                    ]:
                        if field_val.strip():
                            raw_files[modality] = field_val.strip()

                    manifest_data: dict = {
                        "name": ds_name.strip(),
                        "omics": ds_omics,
                        "raw_files": raw_files,
                    }
                    metadata_keys: dict[str, str] = {}
                    if ds_batch_key.strip():
                        metadata_keys["batch"] = ds_batch_key.strip()
                    if ds_cell_type_key.strip():
                        metadata_keys["cell_type"] = ds_cell_type_key.strip()
                    if metadata_keys:
                        manifest_data["metadata_keys"] = metadata_keys

                    with manifest_path.open("w") as fh:
                        yaml.safe_dump(manifest_data, fh, default_flow_style=False, sort_keys=False)

                    ok = _stream_subprocess(
                        [
                            sys.executable, "-m", "multiverse.runner.cli",
                            "register-dataset", "--manifest", str(manifest_path), "--update",
                        ],
                        "Registering dataset…",
                    )
                    if ok:
                        st.session_state["registry_dirty"] = True
                        fetch_registry_data.clear()
        else:
            ds_manifest_path = st.text_input(
                "Path to dataset.yaml", key="ds_manifest_path_direct",
                placeholder="store/datasets/pbmc10k/dataset.yaml",
            )
            if st.button("Register Dataset", key="btn_register_ds_manifest"):
                if not ds_manifest_path.strip():
                    st.error("Manifest path is required.")
                else:
                    ok = _stream_subprocess(
                        [
                            sys.executable, "-m", "multiverse.runner.cli",
                            "register-dataset", "--manifest", ds_manifest_path.strip(), "--update",
                        ],
                        "Registering dataset…",
                    )
                    if ok:
                        st.session_state["registry_dirty"] = True
                        fetch_registry_data.clear()

    # --- Register model ---
    with st.expander("➕ Register Model", expanded=False):
        md_manifest_path = st.text_input(
            "Path to model.yaml", key="md_manifest_path",
            placeholder="store/models/pca/model.yaml",
        )
        build_local = st.toggle("Build Docker image locally", key="md_build_local")

        if st.button("Register Model", key="btn_register_model"):
            if not md_manifest_path.strip():
                st.error("Manifest path is required.")
            else:
                cmd = [
                    sys.executable, "-m", "multiverse.runner.cli",
                    "register-model", "--manifest", md_manifest_path.strip(),
                ]
                if build_local:
                    cmd.append("--build")
                ok = _stream_subprocess(cmd, "Registering model…")
                if ok:
                    st.session_state["registry_dirty"] = True
                    fetch_registry_data.clear()


# ---------------------------------------------------------------------------
# Tab: Job Builder
# ---------------------------------------------------------------------------

def _render_job_builder_tab() -> None:
    import psutil
    import yaml as _yaml

    st.header("Job Builder")

    datasets, models = fetch_registry_data()

    if not datasets:
        st.warning("No datasets found in registry. Register a dataset in the Registry tab first.")
        return

    dataset_name_to_slug = {
        d["name"]: (
            d.get("slug")
            or re.sub(r"[^a-zA-Z0-9._-]+", "-", d["name"]).strip("-").lower()
        )
        for d in datasets
    }

    # --- Read-only compatibility legend ---
    st.subheader("Compatibility Matrix")
    matrix_df = generate_compatibility_matrix(datasets, models)

    def _color_compat(val):
        if val == "Compatible":
            return "background-color: #90ee90; color: #000000"
        if val == "Partial":
            return "background-color: #ffffe0; color: #000000"
        if val == "Incompatible":
            return "background-color: #ffcccb; color: #000000"
        return ""

    st.dataframe(matrix_df.style.map(_color_compat), width='stretch')

    # --- Editable job selection matrix ---
    st.subheader("Select Jobs")
    rows = []
    for ds in datasets:
        for m in models:
            compat = (
                matrix_df.loc[ds["name"], m["name"]]
                if ds["name"] in matrix_df.index and m["name"] in matrix_df.columns
                else "Incompatible"
            )
            rows.append({
                "Selected": compat in ("Compatible", "Partial"),
                "Dataset": ds["name"],
                "Model": m["name"],
                "Compatibility": compat,
            })
    editor_df = pd.DataFrame(rows)

    edited = st.data_editor(
        editor_df,
        column_config={
            "Selected": st.column_config.CheckboxColumn("Selected", default=False),
            "Dataset": st.column_config.TextColumn("Dataset", disabled=True),
            "Model": st.column_config.TextColumn("Model", disabled=True),
            "Compatibility": st.column_config.TextColumn("Compatibility", disabled=True),
        },
        width='stretch',
        hide_index=True,
        key="job_matrix_editor",
    )

    bad = edited["Selected"] & (edited["Compatibility"] == "Incompatible")
    if bad.any():
        st.warning("Incompatible pairs were deselected automatically.")
        edited.loc[bad, "Selected"] = False

    planned_jobs = [
        {"Dataset": row["Dataset"], "Model": row["Model"], "Status": row["Compatibility"]}
        for _, row in edited[edited["Selected"]].iterrows()
    ]
    st.session_state["planned_jobs"] = planned_jobs

    if not planned_jobs:
        st.info("Check rows above to build a job plan.")

    # --- T2.2: Resource summary ---
    if planned_jobs:
        unique_models = {j["Model"] for j in planned_jobs}
        unique_datasets = {j["Dataset"] for j in planned_jobs}

        model_name_to_manifest = {m["name"]: m.get("manifest_path") for m in models}
        committed_gb = 0.0
        for mod_name in unique_models:
            mpath = model_name_to_manifest.get(mod_name)
            if mpath and Path(mpath).exists():
                try:
                    spec = _yaml.safe_load(Path(mpath).read_text())
                    mem_str = spec.get("resources", {}).get("memory_limit", "16g")
                    committed_gb += _parse_memory_gb(mem_str)
                except Exception:
                    committed_gb += 16.0
            else:
                committed_gb += 16.0

        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        delta = avail_gb - committed_gb

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Jobs", len(planned_jobs))
        c2.metric("Unique Datasets", len(unique_datasets))
        c3.metric("Unique Models", len(unique_models))
        c4.metric("Committed RAM", f"{committed_gb:.1f} GB")
        c5.metric("Available RAM", f"{avail_gb:.1f} GB", delta=f"{delta:+.1f} GB")

    # --- Manifest generation ---
    if planned_jobs:
        st.divider()
        st.subheader("Generate Run Manifest")
        experiment_name_input = st.text_input(
            "Experiment Name", value="benchmark_run", key="jb_exp_name"
        )
        random_seed = st.number_input(
            "Random Seed", min_value=0, step=1, value=42, key="jb_seed"
        )
        run_mode = st.radio(
            "Run Mode",
            options=["Use User Params", "Run Gridsearch"],
            index=0,
            horizontal=True,
            key="jb_run_mode",
        )
        st.session_state["run_mode"] = run_mode

        if st.button("Generate Run Manifest", key="btn_gen_manifest"):
            try:
                manifest = build_run_manifest(
                    experiment_name=experiment_name_input,
                    random_seed=int(random_seed),
                    run_mode=run_mode,
                    planned_jobs=planned_jobs,
                    dataset_name_to_slug=dataset_name_to_slug,
                    pair_params=st.session_state.get("pair_params", {}),
                )
            except ValueError as exc:
                st.error(str(exc))
                return

            manifest_path = "run_manifest.yaml"
            with open(manifest_path, "w") as fh:
                yaml.safe_dump(manifest, fh, default_flow_style=False, sort_keys=False)

            st.success(f"Manifest saved to `{manifest_path}`")
            st.code(f"make benchmark config={manifest_path}")
            st.code(
                yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False),
                language="yaml",
            )


# ---------------------------------------------------------------------------
# Tab: Parameters
# ---------------------------------------------------------------------------

def _render_parameters_tab() -> None:
    st.header("Hyperparameter Overrides")

    planned_jobs: list[dict] = st.session_state.get("planned_jobs", [])
    if not planned_jobs:
        st.info("Plan your jobs in the Job Builder tab first.")
        return

    _, models = fetch_registry_data()
    model_name_to_schema_path = {m["name"]: m.get("hyperparameters_schema") for m in models}

    st.caption("Fields are generated from each model's hyperparameter schema.")
    pair_params: dict[tuple[str, str], dict] = {}

    for job in planned_jobs:
        ds_name = job["Dataset"]
        mod_name = job["Model"]
        job_key = f"{ds_name}::{mod_name}"
        field_key = f"params::{job_key}"

        with st.expander(f"{ds_name} × {mod_name}", expanded=False):
            schema = _load_hyperparameter_schema(model_name_to_schema_path.get(mod_name))
            if schema and isinstance(schema.get("properties"), dict):
                pair_params[(ds_name, mod_name)] = render_hyperparameters_form(schema, job_key)
            else:
                st.info("No schema found for this model. Falling back to JSON override input.")
                raw_params = st.text_area(
                    "Model Params (JSON)",
                    value="{}",
                    key=field_key,
                    help="Optional override dictionary passed as model_params for this job.",
                ).strip()
                if not raw_params:
                    raw_params = "{}"
                try:
                    parsed = json.loads(raw_params)
                    if not isinstance(parsed, dict):
                        raise ValueError("Model params must be a JSON object.")
                    pair_params[(ds_name, mod_name)] = parsed
                except Exception as exc:
                    st.error(f"Invalid JSON for {ds_name} × {mod_name}: {exc}")
                    pair_params[(ds_name, mod_name)] = {}

    # Always persist so Job Builder's manifest button can read it
    st.session_state["pair_params"] = pair_params

    # Manifest generation with params
    st.divider()
    if st.button("Generate Run Manifest (with params)", key="btn_gen_manifest_params"):
        datasets, _ = fetch_registry_data()
        dataset_name_to_slug = {
            d["name"]: (
                d.get("slug")
                or re.sub(r"[^a-zA-Z0-9._-]+", "-", d["name"]).strip("-").lower()
            )
            for d in datasets
        }
        run_mode = st.session_state.get("run_mode", "Use User Params")
        try:
            manifest = build_run_manifest(
                experiment_name=st.session_state.get("jb_exp_name", "benchmark_run"),
                random_seed=int(st.session_state.get("jb_seed", 42)),
                run_mode=run_mode,
                planned_jobs=planned_jobs,
                dataset_name_to_slug=dataset_name_to_slug,
                pair_params=pair_params,
            )
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
            return

        manifest_path = "run_manifest.yaml"
        with open(manifest_path, "w") as fh:
            yaml.safe_dump(manifest, fh, default_flow_style=False, sort_keys=False)

        st.success(f"Manifest saved to `{manifest_path}`")
        st.code(
            yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False),
            language="yaml",
        )


# ---------------------------------------------------------------------------
# Tab: Execute  (T4.1 Resource Ledger + T4.2 Live DAG Monitor + T4.3 Live Metrics)
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


# Fragment auto-reruns every 5 s without a full page refresh.
# @st.cache_data(ttl=5) inside fetch_live_metrics deduplicates the MLflow
# API calls across concurrent sessions within each 5-second window.
@st.fragment(run_every=timedelta(seconds=5))
def _live_metrics_panel(experiment_name: str, tracking_uri: str) -> None:
    rows = fetch_live_metrics(experiment_name, tracking_uri)

    if not rows:
        st.info(
            f"No MLflow runs found for experiment **{experiment_name}** yet. "
            "Metrics appear here as each job finishes logging."
        )
        return

    df = pd.DataFrame(rows)

    # Build column config: only include metric columns that have at least one
    # non-empty history list so the table stays compact for sparse experiments.
    col_cfg: dict = {
        "Run": st.column_config.TextColumn("Run", width="medium"),
        "Status": st.column_config.TextColumn("Status", width="small"),
        "Updated": st.column_config.TextColumn("Updated", width="small"),
    }
    cols_to_show = ["Run", "Status", "Updated"]
    for key in LIVE_METRIC_KEYS:
        if key in df.columns and df[key].map(bool).any():
            label = key.upper().replace("_", " ")
            if key in ("ari", "nmi", "silhouette_score"):
                col_cfg[key] = st.column_config.LineChartColumn(
                    label, y_min=0.0, y_max=1.0, width="medium"
                )
            else:
                col_cfg[key] = st.column_config.LineChartColumn(label, width="medium")
            cols_to_show.append(key)

    st.dataframe(
        df[cols_to_show],
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Auto-refreshing every 5 s · {len(rows)} run(s) in experiment **{experiment_name}**"
    )


def _wave_simulate(job_memory: dict[str, float], cap_gb: float) -> list[dict]:
    """Greedy bin-packing wave simulator.  Returns rows for a display table."""
    rows = []
    remaining = cap_gb
    wave = 1
    for job_key, mem_gb in job_memory.items():
        if mem_gb > cap_gb:
            rows.append({"Job": job_key, "RAM (GiB)": f"{mem_gb:.1f}", "Wave": "Too large"})
            continue
        if remaining < mem_gb:
            wave += 1
            remaining = cap_gb
        remaining -= mem_gb
        rows.append({"Job": job_key, "RAM (GiB)": f"{mem_gb:.1f}", "Wave": f"Wave {wave}"})
    return rows


def _render_execute_tab() -> None:
    import psutil

    st.header("Execute")

    planned_jobs: list[dict] = st.session_state.get("planned_jobs", [])
    _, models = fetch_registry_data()
    model_name_to_manifest = {m["name"]: m.get("manifest_path") for m in models}

    # ------------------------------------------------------------------
    # T4.1: Resource Ledger
    # ------------------------------------------------------------------
    st.subheader("Resource Ledger")

    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    used_gb = vm.used / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)

    host_ram_cap = st.number_input(
        "Host RAM Override (GiB)",
        min_value=1.0,
        max_value=float(total_gb),
        value=float(avail_gb),
        step=1.0,
        key="exec_ram_override",
        help="Simulate a smaller machine by reducing the RAM capacity used for admission decisions.",
    )

    # Per-job committed memory
    job_memory: dict[str, float] = {}
    for job in planned_jobs:
        mod_name = job["Model"]
        mem_gb = 16.0
        mpath = model_name_to_manifest.get(mod_name)
        if mpath and Path(mpath).exists():
            try:
                spec = yaml.safe_load(Path(mpath).read_text())
                mem_str = spec.get("resources", {}).get("memory_limit", "16g")
                mem_gb = _parse_memory_gb(mem_str)
            except Exception:
                pass
        job_key = f"{job['Dataset']}_{job['Model']}"
        job_memory[job_key] = mem_gb

    committed_gb = sum(job_memory.values())

    # Progress bars — three bands
    st.caption(f"OS Used: {used_gb:.1f} GiB  |  Committed Jobs: {committed_gb:.1f} GiB  |  Cap: {host_ram_cap:.1f} GiB")
    col_bars = st.columns(1)
    with col_bars[0]:
        st.write(f"**OS Used RAM** — {used_gb:.1f} / {total_gb:.1f} GiB")
        st.progress(min(used_gb / total_gb, 1.0) if total_gb > 0 else 0.0)

        st.write(f"**Committed Job RAM** — {committed_gb:.1f} / {host_ram_cap:.1f} GiB")
        committed_frac = min(committed_gb / host_ram_cap, 1.0) if host_ram_cap > 0 else 0.0
        st.progress(committed_frac)

        free_gb = max(host_ram_cap - committed_gb, 0.0)
        st.write(f"**Free RAM (after jobs)** — {free_gb:.1f} / {host_ram_cap:.1f} GiB")
        st.progress(min(free_gb / host_ram_cap, 1.0) if host_ram_cap > 0 else 0.0)

    if committed_gb > host_ram_cap:
        st.warning(
            f"Committed job RAM ({committed_gb:.1f} GiB) exceeds the host cap "
            f"({host_ram_cap:.1f} GiB). Jobs will be admitted in waves."
        )

    # Wave simulation table
    if job_memory:
        st.subheader("Admission Wave Simulation")
        wave_rows = _wave_simulate(job_memory, host_ram_cap)
        wave_df = pd.DataFrame(wave_rows)
        n_waves = wave_df["Wave"].nunique()
        if n_waves > 1:
            st.info(f"{n_waves} waves needed to fit all jobs within the {host_ram_cap:.1f} GiB cap.")
        st.dataframe(wave_df, width='stretch', hide_index=True)

    st.divider()

    # ------------------------------------------------------------------
    # T4.2: Live DAG Monitor
    # ------------------------------------------------------------------
    st.subheader("Launch & Monitor")

    manifest_path_input = st.text_input(
        "Run Manifest Path",
        value="run_manifest.yaml",
        key="exec_manifest_path",
        help="Manifest generated in the Job Builder or Parameters tab.",
    )
    output_dir_input = st.text_input(
        "Output Directory",
        value="store/artifacts/run_output",
        key="exec_output_dir",
    )
    exec_seed = st.number_input(
        "Random Seed", min_value=0, value=42, step=1, key="exec_seed"
    )

    if not planned_jobs:
        st.warning("No jobs planned. Go to the Job Builder tab first.")

    if st.button("Launch Run", key="btn_launch_run", disabled=not planned_jobs):
        manifest_file = Path(manifest_path_input.strip())
        if not manifest_file.exists():
            st.error(
                f"Manifest not found: `{manifest_file}`. "
                "Generate it in the Job Builder or Parameters tab first."
            )
        else:
            cmd = [
                sys.executable, "-m", "multiverse.runner.cli", "run",
                "--manifest", str(manifest_file),
                "--output", output_dir_input.strip(),
                "--seed", str(int(exec_seed)),
            ]

            # Build initial status table keyed by CLI job name format
            job_statuses: dict[str, str] = {
                f"{j['Dataset']}_{j['Model']}": "⏳ Pending"
                for j in planned_jobs
            }
            status_placeholder = st.empty()

            def _refresh_status_table():
                status_placeholder.dataframe(
                    pd.DataFrame(
                        [{"Job": k, "Status": v} for k, v in job_statuses.items()]
                    ),
                    width='stretch',
                    hide_index=True,
                )

            _refresh_status_table()

            with st.status("Running pipeline…", expanded=True) as run_status:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for raw_line in proc.stdout:
                    clean = _ANSI_RE.sub("", raw_line).rstrip()
                    if not clean:
                        continue
                    st.write(clean)

                    lc = clean.lower()
                    for job_key in job_statuses:
                        if job_key not in clean:
                            continue
                        if any(w in lc for w in ("starting", "admitted", "running")):
                            job_statuses[job_key] = "🔵 Training"
                        elif any(w in lc for w in ("success", "completed", "promoted")):
                            job_statuses[job_key] = "🟢 Done"
                        elif any(w in lc for w in ("failed", "error", "insufficient")):
                            job_statuses[job_key] = "🔴 Failed"
                    _refresh_status_table()

                proc.wait()
                if proc.returncode == 0:
                    run_status.update(label="Pipeline completed successfully ✓", state="complete")
                    for k, v in job_statuses.items():
                        if v == "⏳ Pending":
                            job_statuses[k] = "🟢 Done"
                else:
                    run_status.update(
                        label=f"Pipeline exited with code {proc.returncode} ✗",
                        state="error",
                    )
                    for k, v in job_statuses.items():
                        if v in ("⏳ Pending", "🔵 Training"):
                            job_statuses[k] = "🔴 Failed"
                _refresh_status_table()

    # ------------------------------------------------------------------
    # T4.3: Live MLflow Metrics
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Live MLflow Metrics")

    mlflow_base = _get_mlflow_url()
    if not _check_service(f"{mlflow_base}/health"):
        st.info("MLflow is offline — start it with `make services-up` to see live metrics.")
    else:
        # Default to the experiment name used in the most recent manifest
        saved_exp = st.session_state.get("active_experiment_name", "")
        jb_exp_raw = st.session_state.get("jb_exp_name", "benchmark_run")
        try:
            jb_exp_slug = slugify_experiment_name(jb_exp_raw)
        except ValueError:
            jb_exp_slug = "benchmark_run"
        default_exp = saved_exp or jb_exp_slug

        exp_col, _ = st.columns([2, 3])
        with exp_col:
            monitor_exp = st.text_input(
                "Experiment to monitor",
                value=default_exp,
                key="exec_live_exp_name",
                help="MLflow experiment name. Auto-populated from the Job Builder manifest.",
            )

        if monitor_exp.strip():
            _live_metrics_panel(monitor_exp.strip(), mlflow_base)


# ---------------------------------------------------------------------------
# Tab: Results (stub)
# ---------------------------------------------------------------------------

def _fetch_runs(status_filter: str | None = None) -> list[dict]:
    """Query the runs table, optionally filtered by status."""
    init_db()
    conn = get_db_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if status_filter:
            cursor.execute(
                "SELECT run_id, dataset_id, model_name, status, output_path "
                "FROM runs WHERE status = ? ORDER BY run_id DESC",
                (status_filter,),
            )
        else:
            cursor.execute(
                "SELECT run_id, dataset_id, model_name, status, output_path "
                "FROM runs ORDER BY run_id DESC"
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict for display."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _render_results_tab() -> None:
    st.header("Results")

    col_filter, col_refresh = st.columns([4, 1])
    with col_filter:
        status_choice = st.selectbox(
            "Filter by status",
            options=["All", "SUCCESS", "FAILED", "RUNNING"],
            key="results_status_filter",
        )
    with col_refresh:
        st.write("")
        if st.button("Refresh", key="btn_results_refresh"):
            st.cache_data.clear()

    filter_val = None if status_choice == "All" else status_choice
    runs = _fetch_runs(filter_val)

    if not runs:
        st.info("No runs found. Launch a benchmarking run from the Execute tab first.")
        return

    # Summary table
    summary_rows = [
        {
            "Run ID": r["run_id"],
            "Model": r["model_name"] or "—",
            "Status": r["status"],
            "Output Path": r["output_path"] or "—",
        }
        for r in runs
    ]
    st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)

    # Drill-down selector
    success_runs = [r for r in runs if r["status"] == "SUCCESS"]
    if not success_runs:
        st.info("No SUCCESS runs to inspect yet.")
        return

    st.subheader("Drill Down")
    run_labels = [
        f"Run {r['run_id']} — {r['model_name'] or 'unknown'}"
        for r in success_runs
    ]
    selected_label = st.selectbox("Select a run", options=run_labels, key="results_run_selector")
    selected_run = success_runs[run_labels.index(selected_label)]
    artifact_dir = Path(selected_run["output_path"]) if selected_run["output_path"] else None

    if not artifact_dir or not artifact_dir.exists():
        st.warning(f"Artifact directory not found: `{artifact_dir}`")
        return

    # Metrics
    metrics_file = artifact_dir / "metrics.json"
    if metrics_file.exists():
        try:
            raw_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            flat = _flatten_dict(raw_metrics)
            metrics_df = pd.DataFrame(
                [{"Metric": k, "Value": v} for k, v in flat.items() if isinstance(v, (int, float))]
            )
            st.subheader("Metrics")
            if not metrics_df.empty:
                col_table, col_chart = st.columns([1, 1])
                with col_table:
                    st.dataframe(metrics_df, width='stretch', hide_index=True)
                with col_chart:
                    chart_df = metrics_df.set_index("Metric")["Value"]
                    st.bar_chart(chart_df)
            else:
                st.json(raw_metrics)
        except Exception as exc:
            st.error(f"Could not parse metrics.json: {exc}")
    else:
        st.info("No metrics.json found in artifact directory.")

    # Container log
    log_file = artifact_dir / "container.log"
    with st.expander("Container Log", expanded=False):
        if log_file.exists():
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            st.text(log_text if log_text.strip() else "(empty log)")
        else:
            st.info("No container.log found.")

    # Provenance note
    job_spec_file = artifact_dir / "job_spec.json"
    st.subheader("Provenance")
    st.caption(f"Artifact directory: `{artifact_dir}`")
    if job_spec_file.exists():
        st.caption(f"Job spec: `{job_spec_file}`")
        with st.expander("Job Spec", expanded=False):
            try:
                st.json(json.loads(job_spec_file.read_text(encoding="utf-8")))
            except Exception:
                st.text(job_spec_file.read_text(encoding="utf-8", errors="replace"))

    # MLflow contextual routing — auto-detect experiment name from job_spec, then
    # resolve to an MLflow experiment ID so the 🔬 tab can deep-link directly.
    st.divider()
    st.subheader("MLflow Deep-Link")
    mlflow_base = _get_mlflow_url()
    mlflow_live = _check_service(f"{mlflow_base}/health")

    if not mlflow_live:
        st.caption("MLflow is offline — start it with `make services-up` to enable deep-linking.")
    else:
        # Try auto-detection from job_spec.json
        auto_exp_name: str | None = None
        if job_spec_file.exists():
            try:
                spec_data = json.loads(job_spec_file.read_text(encoding="utf-8"))
                auto_exp_name = (
                    spec_data.get("run_settings", {}).get("experiment_name")
                    or spec_data.get("globals", {}).get("experiment_name")
                    or spec_data.get("experiment_name")
                )
            except Exception:
                pass

        if auto_exp_name:
            exp_id = _resolve_mlflow_experiment_id(auto_exp_name, mlflow_base)
            if exp_id:
                st.session_state["active_experiment_id"] = exp_id
                st.session_state["active_experiment_name"] = auto_exp_name
                st.success(
                    f"Active experiment set to **{auto_exp_name}** (ID: `{exp_id}`). "
                    "Switch to the 🔬 Experiment Analysis tab to view it."
                )
            else:
                st.info(
                    f"Experiment `{auto_exp_name}` not found in MLflow yet "
                    "(run may not have been tracked)."
                )

        # Always offer manual override
        with st.expander("Set experiment manually", expanded=auto_exp_name is None):
            manual_exp = st.text_input(
                "MLflow experiment name",
                value=st.session_state.get("active_experiment_name", ""),
                key="manual_exp_name",
                placeholder="benchmark_run",
            )
            col_set, col_clear = st.columns([2, 1])
            with col_set:
                if st.button("Set active experiment", key="btn_set_exp"):
                    name = manual_exp.strip()
                    if name:
                        exp_id = _resolve_mlflow_experiment_id(name, mlflow_base)
                        if exp_id:
                            st.session_state["active_experiment_id"] = exp_id
                            st.session_state["active_experiment_name"] = name
                            st.success(f"Linked to experiment '{name}' (ID: {exp_id})")
                        else:
                            st.warning(f"Experiment '{name}' not found in MLflow.")
            with col_clear:
                if st.button("Clear", key="btn_clear_exp_results"):
                    st.session_state["active_experiment_id"] = None
                    st.session_state["active_experiment_name"] = ""
                    st.rerun()


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------

def _check_service(url: str, timeout: float = 1.5) -> bool:
    """Return True if the HTTP service responds with a non-5xx status."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def _get_mlflow_url() -> str:
    url = os.environ.get("MLFLOW_UI_URL", "") or os.environ.get("MLFLOW_TRACKING_URI", "")
    if not url.startswith("http"):
        url = "http://localhost:5000"
    return url.rstrip("/")


def _get_optuna_url() -> str:
    url = os.environ.get("OPTUNA_UI_URL", "")
    if not url.startswith("http"):
        url = f"http://localhost:{os.environ.get('OPTUNA_PORT', '8080')}"
    return url.rstrip("/")


def _resolve_mlflow_experiment_id(experiment_name: str, mlflow_base: str) -> str | None:
    """Call MLflow REST API to map experiment name → ID. No mlflow package import."""
    import urllib.parse
    encoded = urllib.parse.quote(experiment_name, safe="")
    api_url = f"{mlflow_base}/api/2.0/mlflow/experiments/get-by-name?experiment_name={encoded}"
    try:
        with urllib.request.urlopen(api_url, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            return str(data["experiment"]["experiment_id"])
    except Exception:
        return None


def _render_observability_sidebar() -> None:
    mlflow_base = _get_mlflow_url()
    optuna_base = _get_optuna_url()
    mlflow_up = _check_service(f"{mlflow_base}/health")
    optuna_up = _check_service(optuna_base)

    with st.sidebar:
        st.subheader("Observability Services")
        col_mf, col_op = st.columns(2)
        with col_mf:
            if mlflow_up:
                st.success("MLflow")
                st.markdown(f"[Open]({mlflow_base})")
            else:
                st.error("MLflow offline")
                st.caption("`make services-up`")
        with col_op:
            if optuna_up:
                st.success("Optuna")
                st.markdown(f"[Open]({optuna_base})")
            else:
                st.error("Optuna offline")
                st.caption("`make services-up`")


# ---------------------------------------------------------------------------
# Tab: Experiment Analysis (MLflow)
# ---------------------------------------------------------------------------

def _render_mlflow_tab() -> None:
    mlflow_base = _get_mlflow_url()
    mlflow_up = _check_service(f"{mlflow_base}/health")

    st.header("Experiment Analysis")

    if not mlflow_up:
        st.warning(
            "MLflow Tracking Server is not reachable. "
            "Start it with `make services-up`, then refresh this tab."
        )
        st.link_button("Open MLflow UI (new tab)", mlflow_base)
        return

    exp_id = st.session_state.get("active_experiment_id")
    if exp_id:
        exp_name = st.session_state.get("active_experiment_name", exp_id)
        deep_url = f"{mlflow_base}/#/experiments/{exp_id}"
        col_info, col_clear = st.columns([4, 1])
        with col_info:
            st.info(f"Showing experiment **{exp_name}** — ID `{exp_id}`")
        with col_clear:
            if st.button("Show all", key="btn_mlflow_show_all"):
                st.session_state["active_experiment_id"] = None
                st.session_state["active_experiment_name"] = ""
                st.rerun()
    else:
        deep_url = mlflow_base
        st.caption(
            "No active experiment selected. "
            "Pick a run in the Results tab to deep-link here automatically."
        )

    col_link, col_h = st.columns([3, 1])
    with col_link:
        st.link_button("Open in new tab", deep_url)
    with col_h:
        iframe_h = st.slider("Height (px)", 400, 1200, 860, step=50, key="mlflow_iframe_h")

    st.caption(
        "If the frame appears blank your browser may be blocking mixed content "
        "(HTTP iframe inside an HTTPS page). Use the button above to open in a new tab. "
        "For remote deployments, front both services through a TLS-terminating reverse proxy."
    )
    components.iframe(deep_url, height=iframe_h, scrolling=True)


# ---------------------------------------------------------------------------
# Tab: Sweep Tracker (Optuna)
# ---------------------------------------------------------------------------

def _render_optuna_tab() -> None:
    optuna_base = _get_optuna_url()
    optuna_up = _check_service(optuna_base)

    st.header("Sweep Tracker")

    if not optuna_up:
        st.warning(
            "Optuna Dashboard is not reachable. "
            "Start it with `make services-up`, then refresh this tab."
        )
        st.link_button("Open Optuna Dashboard (new tab)", optuna_base)
        return

    col_link, col_h = st.columns([3, 1])
    with col_link:
        st.link_button("Open in new tab", optuna_base)
    with col_h:
        iframe_h = st.slider("Height (px)", 400, 1200, 860, step=50, key="optuna_iframe_h")

    st.caption(
        "If the frame appears blank your browser may be blocking mixed content. "
        "Use the button above to open in a new tab."
    )
    components.iframe(optuna_base, height=iframe_h, scrolling=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Multiverse", layout="wide")
    _init_session_state()
    _render_observability_sidebar()

    st.title("Multiverse Benchmarking Platform")

    tab_registry, tab_jobs, tab_params, tab_execute, tab_results, tab_mlflow, tab_optuna = st.tabs(
        [
            "📦 Registry",
            "🧬 Job Builder",
            "⚙️ Parameters",
            "🚀 Execute",
            "📊 Results",
            "🔬 Experiment Analysis",
            "📈 Sweep Tracker",
        ]
    )

    with tab_registry:
        _render_registry_tab()

    with tab_jobs:
        _render_job_builder_tab()

    with tab_params:
        _render_parameters_tab()

    with tab_execute:
        _render_execute_tab()

    with tab_results:
        _render_results_tab()

    with tab_mlflow:
        _render_mlflow_tab()

    with tab_optuna:
        _render_optuna_tab()


if __name__ == "__main__":
    main()
