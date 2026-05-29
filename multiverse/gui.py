import html
import json
import os
import re
import queue
import shutil
import sqlite3
import subprocess
import threading
import sys
import time
import urllib.request
from collections import deque
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # type: ignore[import-untyped]
import yaml
from multiverse.gui_artifacts import render_artifact_tree, render_download_button, render_log_viewer
from multiverse.gui_navigation import go_to, render_top_nav
from multiverse.gui_telemetry import track
from multiverse.gui_state import bump_editor_version, get_state, init_state
from multiverse.gui_utils import LIVE_METRIC_KEYS, fetch_live_metrics, render_hyperparameters_form
from multiverse.multiverse_config import (
    DEFAULT_DOCKER_DATA_ROOT,
    get_config,
    get_docker_data_root,
    save_config,
)
from multiverse.registry import generate_compatibility_matrix
from multiverse.registry_db import (
    get_all_datasets,
    get_all_models,
    get_db_connection,
    init_db,
    mark_dataset_removed,
    mark_model_inactive,
)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data
def fetch_registry_data():
    init_db()
    datasets = [d for d in get_all_datasets() if d.get("status") != "REMOVED"]
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
    selected_pairs = {(job["Dataset"], job["Model"]) for job in planned_jobs}
    scoped_pair_params = {
        pair: params
        for pair, params in pair_params.items()
        if pair in selected_pairs and isinstance(params, dict)
    }
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
                "model_params": scoped_pair_params.get((ds_name, mod_name), {}) or {},
            }
        )
    return manifest


def render_manifest_errors(errors: list[dict], *, title: str = "Manifest validation failed") -> None:
    st.error(title)
    if not errors:
        return
    for err in errors:
        code = err.get("code", "invalid")
        field = err.get("field", "manifest")
        message = err.get("message", "")
        with st.expander(f"{code}: {field}", expanded=True):
            st.write(message)
            st.caption("Fix the manifest or registry entry, then launch again.")
    st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)


def paginate(total_count_fn, page_fn, page_size: int = 50, key: str = "page") -> list[dict]:
    total = int(total_count_fn())
    n_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, key=key)
    rows = page_fn(limit=page_size, offset=(int(page) - 1) * page_size)
    st.caption(f"Page {page} of {n_pages} · {total} total rows")
    return rows


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

def _generate_example_dataset() -> Path:
    """Write a small synthetic AnnData + manifest under store/datasets/example.

    Lazy import of anndata + numpy keeps the GUI startup cheap; the helper is
    only reached when the user clicks 'Load Example Dataset' in the empty
    Registry tab.  Returns the manifest path so the caller can invoke the
    register-dataset CLI on it.
    """
    import anndata as ad
    import numpy as np

    target_dir = Path("store/datasets/example")
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)

    n_cells, n_genes = 50, 200
    counts = rng.poisson(lam=0.5, size=(n_cells, n_genes)).astype("float32")
    obs_names = [f"cell_{i:03d}" for i in range(n_cells)]
    var_names = [f"gene_{i:04d}" for i in range(n_genes)]
    adata = ad.AnnData(X=counts)
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obs["batch"] = ["A" if i % 2 == 0 else "B" for i in range(n_cells)]
    adata.obs["cell_type"] = [f"type_{i % 3}" for i in range(n_cells)]
    data_path = target_dir / "rna.h5ad"
    adata.write_h5ad(data_path)

    manifest_path = target_dir / "dataset.yaml"
    yaml.safe_dump(
        {
            "name": "Example Synthetic",
            "omics": ["rna"],
            "raw_files": {"rna": str(data_path.resolve())},
            "metadata_keys": {"batch": "batch", "cell_type": "cell_type"},
        },
        manifest_path.open("w"),
        default_flow_style=False,
        sort_keys=False,
    )
    return manifest_path


def _render_registry_welcome() -> None:
    """Empty-state welcome panel for first-time users (O-01 + N-09)."""
    st.success("Welcome to Multiverse! Let's get you to your first benchmark run.")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "**1. Register** &nbsp;\n\n"
        "Add a dataset (an `.h5ad` or `.h5mu` file) and the models you want to compare."
    )
    c2.markdown(
        "**2. Plan** &nbsp;\n\n"
        "Go to **Configure** to pick dataset × model pairs from the compatibility matrix."
    )
    c3.markdown(
        "**3. Execute** &nbsp;\n\n"
        "Hit **Launch Run** on the Run tab and watch metrics stream into the Results tab."
    )
    st.divider()
    st.markdown("**Try it without a real dataset:**")
    if st.button("Load Example Dataset", key="btn_load_example_dataset"):
        try:
            manifest_path = _generate_example_dataset()
        except Exception as exc:
            st.error(f"Could not generate example dataset: {exc}")
            return
        ok = _stream_subprocess(
            [
                sys.executable, "-m", "multiverse.runner.cli",
                "register-dataset", "--manifest", str(manifest_path), "--update",
            ],
            "Registering example dataset...",
        )
        if ok:
            track("example_dataset_loaded")
            st.session_state["registry_dirty"] = True
            fetch_registry_data.clear()
            st.rerun()


def _render_registry_tab() -> None:
    st.header("Asset Registry")

    if st.session_state.get("registry_dirty"):
        st.warning("Registry changed. Refresh the registry data before building or launching new jobs.")
        if st.button("Refresh Registry Data", key="btn_refresh_registry_dirty"):
            track("registry_refreshed")
            fetch_registry_data.clear()
            st.session_state["registry_dirty"] = False
            st.rerun()
    elif st.button("Refresh Registry", key="btn_refresh_registry"):
        track("registry_refreshed")
        fetch_registry_data.clear()
        st.rerun()

    datasets, models = fetch_registry_data()

    if not datasets and not models:
        _render_registry_welcome()
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
                st.dataframe(
                    pd.DataFrame(ds_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "slug": st.column_config.TextColumn("Slug", width="medium"),
                        "name": st.column_config.TextColumn("Name", width="medium"),
                        "omics": st.column_config.TextColumn("Omics", width="small"),
                        "status": st.column_config.TextColumn("Status", width="small"),
                    },
                )
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
                st.dataframe(
                    pd.DataFrame(md_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "slug": st.column_config.TextColumn("Slug", width="medium"),
                        "version": st.column_config.TextColumn("Version", width="small"),
                        "name": st.column_config.TextColumn("Name", width="medium"),
                        "omics": st.column_config.TextColumn("Omics", width="small"),
                        "status": st.column_config.TextColumn("Status", width="small"),
                    },
                )
            else:
                st.caption("No models registered.")


        if datasets or models:
            st.divider()
            col_rm_ds, col_rm_md = st.columns(2)
            with col_rm_ds:
                st.subheader("Remove Dataset")
                if datasets:
                    ds_options = {f"{d.get('name', d.get('slug'))} ({d.get('slug')})": d for d in datasets}
                    ds_label = st.selectbox("Dataset", options=list(ds_options), key="remove_dataset_choice")
                    ds = ds_options[ds_label]
                    if st.button("Remove dataset", key="btn_remove_dataset"):
                        st.session_state["confirm_remove_dataset"] = ds.get("slug") or ds.get("id")
                    if st.session_state.get("confirm_remove_dataset") == (ds.get("slug") or ds.get("id")):
                        st.warning("This hides the dataset from new jobs but keeps historical runs.")
                        if st.button("Confirm remove dataset", key="btn_confirm_remove_dataset"):
                            if mark_dataset_removed(ds.get("slug") or ds.get("id")):
                                track("dataset_removed", slug=ds.get("slug"))
                                fetch_registry_data.clear()
                                st.session_state.pop("confirm_remove_dataset", None)
                                st.session_state["registry_dirty"] = True
                                st.rerun()
                            else:
                                st.error("Dataset was not found.")
                else:
                    st.caption("No datasets registered.")
            with col_rm_md:
                st.subheader("Remove Model")
                if models:
                    md_options = {f"{m.get('name', m.get('slug'))} {m.get('version', '')} ({m.get('slug')})": m for m in models}
                    md_label = st.selectbox("Model", options=list(md_options), key="remove_model_choice")
                    md = md_options[md_label]
                    confirm_key = f"{md.get('slug')}::{md.get('version')}"
                    if st.button("Remove model", key="btn_remove_model"):
                        st.session_state["confirm_remove_model"] = confirm_key
                    if st.session_state.get("confirm_remove_model") == confirm_key:
                        st.warning("This hides the model from new jobs but keeps historical runs.")
                        if st.button("Confirm remove model", key="btn_confirm_remove_model"):
                            if mark_model_inactive(str(md.get("slug")), md.get("version")):
                                track("model_removed", slug=md.get("slug"), version=md.get("version"))
                                fetch_registry_data.clear()
                                st.session_state.pop("confirm_remove_model", None)
                                st.session_state["registry_dirty"] = True
                                st.rerun()
                            else:
                                st.error("Model was not found.")
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

            if st.button("Register from fields", key="btn_register_ds_fields"):
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
                        track("dataset_registered", source="fields", manifest_path=str(manifest_path))
                        st.session_state["registry_dirty"] = True
                        fetch_registry_data.clear()
                        st.rerun()
        else:
            ds_manifest_path = st.text_input(
                "Path to dataset.yaml", key="ds_manifest_path_direct",
                placeholder="store/datasets/pbmc10k/dataset.yaml",
            )
            if st.button("Register from manifest", key="btn_register_ds_manifest"):
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
                        track("dataset_registered", source="manifest", manifest_path=ds_manifest_path.strip())
                        st.session_state["registry_dirty"] = True
                        fetch_registry_data.clear()
                        st.rerun()

    # --- Register model ---
    with st.expander("Register Model", expanded=False):
        use_model_fields = st.toggle(
            "Build manifest from fields (I don't have a model.yaml yet)",
            key="md_use_fields",
        )
        build_local = st.toggle("Build Docker image locally", key="md_build_local")

        if use_model_fields:
            md_name = st.text_input("Model name", key="md_name")
            md_version = st.text_input("Version", value="0.1.0", key="md_version")
            md_image = st.text_input(
                "Runtime image tag",
                key="md_image",
                placeholder="ghcr.io/org/model:0.1.0",
            )
            md_omics = st.multiselect(
                "Supported omics",
                options=["rna", "atac", "adt", "any"],
                key="md_omics",
            )
            if st.button("Register from fields", key="btn_register_model_fields"):
                if not md_name.strip():
                    st.error("Model name is required.")
                elif not md_version.strip():
                    st.error("Version is required.")
                elif not md_image.strip():
                    st.error("Runtime image tag is required.")
                elif not md_omics:
                    st.error("Select at least one supported omics modality.")
                elif "any" in md_omics and len(md_omics) > 1:
                    st.error("Use `any` by itself, or select specific modalities.")
                else:
                    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", md_name.strip()).strip("-").lower()
                    manifest_dir = Path("store/models") / slug
                    manifest_dir.mkdir(parents=True, exist_ok=True)
                    manifest_path = manifest_dir / "model.yaml"
                    manifest_data = {
                        "name": md_name.strip(),
                        "version": md_version.strip(),
                        "supported_omics": md_omics,
                        "runtime": {"image": md_image.strip()},
                    }
                    with manifest_path.open("w") as fh:
                        yaml.safe_dump(manifest_data, fh, default_flow_style=False, sort_keys=False)
                    cmd = [
                        sys.executable, "-m", "multiverse.runner.cli",
                        "register-model", "--manifest", str(manifest_path),
                    ]
                    if build_local:
                        cmd.append("--build")
                    ok = _stream_subprocess(cmd, "Registering model...")
                    if ok:
                        track("model_registered", source="fields", manifest_path=str(manifest_path), build_local=build_local)
                        st.session_state["registry_dirty"] = True
                        fetch_registry_data.clear()
                        st.rerun()
        else:
            md_manifest_path = st.text_input(
                "Path to model.yaml", key="md_manifest_path",
                placeholder="store/models/pca/model.yaml",
            )
            if st.button("Register from manifest", key="btn_register_model"):
                if not md_manifest_path.strip():
                    st.error("Manifest path is required.")
                else:
                    cmd = [
                        sys.executable, "-m", "multiverse.runner.cli",
                        "register-model", "--manifest", md_manifest_path.strip(),
                    ]
                    if build_local:
                        cmd.append("--build")
                    ok = _stream_subprocess(cmd, "Registering model...")
                    if ok:
                        track("model_registered", source="manifest", manifest_path=md_manifest_path.strip(), build_local=build_local)
                        st.session_state["registry_dirty"] = True
                        fetch_registry_data.clear()
                        st.rerun()


# ---------------------------------------------------------------------------
# Tab: Configure
# ---------------------------------------------------------------------------

def _parse_manifest_job_selection(loaded: dict) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    jobs = loaded.get("jobs", []) if isinstance(loaded, dict) else []
    staged_jobs: list[dict] = []
    staged_params: dict[tuple[str, str], dict] = {}
    if not isinstance(jobs, list):
        return staged_jobs, staged_params
    for item in jobs:
        if not isinstance(item, dict):
            continue
        dataset_slug = str(item.get("dataset_slug") or "").strip()
        model_name = str(item.get("model_name") or "").strip()
        if not dataset_slug or not model_name:
            continue
        staged_jobs.append({"dataset_slug": dataset_slug, "model_name": model_name})
        params = item.get("model_params", {})
        if isinstance(params, dict):
            staged_params[(dataset_slug, model_name)] = dict(params)
    return staged_jobs, staged_params


def _stage_loaded_manifest(loaded: dict, load_path: str) -> None:
    globals_cfg = loaded.get("globals", {}) if isinstance(loaded, dict) else {}
    if "experiment_name" in globals_cfg:
        st.session_state["_pending_shared_experiment_name"] = str(globals_cfg["experiment_name"])
    if "random_seed" in globals_cfg:
        st.session_state["_pending_shared_seed"] = int(globals_cfg["random_seed"])
    if globals_cfg.get("run_gridsearch"):
        st.session_state["_pending_shared_run_mode"] = "Run Gridsearch"
    elif globals_cfg.get("run_user_params"):
        st.session_state["_pending_shared_run_mode"] = "Use User Params"
    staged_jobs, staged_params = _parse_manifest_job_selection(loaded)
    st.session_state["_pending_manifest_jobs"] = staged_jobs
    st.session_state["_pending_manifest_pair_params"] = staged_params
    st.session_state["_pending_shared_manifest_path"] = load_path.strip() or "run_manifest.yaml"
    count = len(staged_jobs)
    noun = "job" if count == 1 else "jobs"
    st.session_state["_manifest_load_notice"] = f"Manifest settings loaded ({count} {noun})."


def _apply_pending_manifest_jobs(
    datasets: list[dict],
    models: list[dict],
) -> dict[tuple[str, str], dict]:
    staged_jobs = st.session_state.pop("_pending_manifest_jobs", None)
    staged_params = st.session_state.pop("_pending_manifest_pair_params", {})
    if not staged_jobs:
        return {}

    dataset_slug_to_name = {
        str(d.get("slug") or re.sub(r"[^a-zA-Z0-9._-]+", "-", d.get("name", "")).strip("-").lower()): d.get("name")
        for d in datasets
    }
    model_name_lookup = {str(m.get("name")): str(m.get("name")) for m in models}
    for m in models:
        name = str(m.get("name"))
        slug = str(m.get("slug") or "")
        model_name_lookup.setdefault(name.lower(), name)
        if slug:
            model_name_lookup.setdefault(slug, name)
            model_name_lookup.setdefault(slug.lower(), name)
    selected_datasets: list[str] = []
    selected_models: list[str] = []
    loaded_pair_params: dict[tuple[str, str], dict] = {}

    for job in staged_jobs:
        if not isinstance(job, dict):
            continue
        dataset_slug = str(job.get("dataset_slug") or "")
        requested_model_name = str(job.get("model_name") or "")
        model_name = model_name_lookup.get(requested_model_name) or model_name_lookup.get(requested_model_name.lower())
        dataset_name = dataset_slug_to_name.get(dataset_slug)
        if not dataset_name or not model_name:
            continue
        if dataset_name not in selected_datasets:
            selected_datasets.append(dataset_name)
        if model_name not in selected_models:
            selected_models.append(model_name)
        params = staged_params.get((dataset_slug, requested_model_name), {})
        if isinstance(params, dict):
            loaded_pair_params[(dataset_name, model_name)] = dict(params)

    if selected_datasets:
        st.session_state["selected_datasets"] = selected_datasets
    if selected_models:
        st.session_state["selected_models"] = selected_models
    if loaded_pair_params:
        st.session_state["_loaded_manifest_pair_params"] = loaded_pair_params
    bump_editor_version()
    st.session_state.pop("job_matrix_signature", None)
    return loaded_pair_params


def _prefill_hyperparameter_widget_state(job_key: str, params: dict) -> None:
    for param_name, value in params.items():
        if isinstance(value, dict) and value.get("type") in {"int", "float", "categorical"}:
            st.session_state[f"{job_key}::sweep_toggle::{param_name}"] = True
            base_key = f"{job_key}::sweep::{param_name}"
            if value.get("type") == "categorical":
                st.session_state[f"{base_key}::choices"] = value.get("choices", [])
            else:
                if "low" in value:
                    st.session_state[f"{base_key}::low"] = value["low"]
                if "high" in value:
                    st.session_state[f"{base_key}::high"] = value["high"]
                st.session_state[f"{base_key}::dist"] = (
                    "int_log_uniform" if value.get("type") == "int" and value.get("log")
                    else "int_uniform" if value.get("type") == "int"
                    else "float_log_uniform" if value.get("log")
                    else "float_uniform"
                )
            continue
        st.session_state.setdefault(f"{job_key}::sweep_toggle::{param_name}", False)
        st.session_state[f"{job_key}::fixed::{param_name}"] = value


def _consume_loaded_pair_params(ds_name: str, mod_name: str) -> dict:
    loaded = st.session_state.get("_loaded_manifest_pair_params", {})
    if not isinstance(loaded, dict):
        return {}
    params = loaded.pop((ds_name, mod_name), {})
    if not loaded:
        st.session_state.pop("_loaded_manifest_pair_params", None)
    return params if isinstance(params, dict) else {}



def _apply_pending_shared_config() -> None:
    pending_map = {
        "_pending_shared_experiment_name": "shared_experiment_name",
        "_pending_shared_seed": "shared_seed",
        "_pending_shared_run_mode": "shared_run_mode",
        "_pending_shared_manifest_path": "shared_manifest_path",
    }
    for pending_key, target_key in pending_map.items():
        if pending_key in st.session_state:
            st.session_state[target_key] = st.session_state.pop(pending_key)
    st.session_state["experiment_name"] = st.session_state.get("shared_experiment_name", "benchmark_run")
    st.session_state["run_mode"] = st.session_state.get("shared_run_mode", "Use User Params")


def _render_load_manifest_panel() -> None:
    st.subheader("Load Existing Manifest")
    st.info("Import global settings from an existing run manifest before selecting jobs.")
    load_path = st.text_input(
        "Manifest to load",
        value=st.session_state.get("shared_manifest_path", "run_manifest.yaml"),
        key="configure_load_manifest_path",
    )
    if st.button("Load Manifest Settings", key="btn_load_manifest_settings"):
        try:
            loaded = yaml.safe_load(Path(load_path).read_text(encoding="utf-8")) or {}
            _stage_loaded_manifest(loaded, load_path)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load manifest: {exc}")


def _render_manifest_load_notice() -> None:
    notice = st.session_state.pop("_manifest_load_notice", None)
    if notice:
        st.success(str(notice))


def _render_run_configuration() -> tuple[str, int, str, str]:
    st.subheader("Run Configuration")
    experiment_name_input = st.text_input(
        "Experiment Name",
        value=st.session_state.get("shared_experiment_name", "benchmark_run"),
        key="shared_experiment_name",
        help="Used for manifest generation and live metrics.",
    )
    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        step=1,
        value=int(st.session_state.get("shared_seed", 42)),
        key="shared_seed",
    )
    run_mode = st.radio(
        "Run Mode",
        options=["Use User Params", "Run Gridsearch"],
        index=0 if st.session_state.get("shared_run_mode", "Use User Params") == "Use User Params" else 1,
        horizontal=True,
        key="shared_run_mode",
    )
    st.session_state["run_mode"] = run_mode
    st.session_state["experiment_name"] = experiment_name_input or "benchmark_run"
    manifest_path = st.text_input(
        "Manifest save path",
        value=st.session_state.get("shared_manifest_path", "run_manifest.yaml"),
        key="shared_manifest_path",
    )
    return experiment_name_input or "benchmark_run", int(random_seed), run_mode, manifest_path


def _render_configure_tab() -> None:
    _apply_pending_shared_config()

    st.header("Configure")
    _render_load_manifest_panel()
    _render_manifest_load_notice()
    st.divider()
    experiment_name_input, random_seed, run_mode, manifest_path = _render_run_configuration()
    st.divider()

    datasets, models = fetch_registry_data()

    if not datasets:
        st.warning("No datasets found in registry. Register a dataset in the Registry tab first.")
        if st.button("Go to Registry →", key="shortcut_configure_registry"):
            go_to("registry")
        return
    if not models:
        st.warning("No models found in registry. Register a model in the Registry tab first.")
        if st.button("Go to Registry →", key="shortcut_configure_registry_models"):
            go_to("registry")
        return

    loaded_pair_params = _apply_pending_manifest_jobs(datasets, models)

    dataset_name_to_slug = {
        d["name"]: (
            d.get("slug")
            or re.sub(r"[^a-zA-Z0-9._-]+", "-", d["name"]).strip("-").lower()
        )
        for d in datasets
    }

    st.subheader("1. Select Jobs")
    matrix_df = generate_compatibility_matrix(datasets, models)

    dataset_names = [d["name"] for d in datasets]
    model_names = [m["name"] for m in models]
    default_datasets = dataset_names if len(dataset_names) <= 10 else []
    default_models = model_names if len(model_names) <= 10 else []
    previous_datasets = [name for name in st.session_state.get("selected_datasets", []) if name in dataset_names]
    previous_models = [name for name in st.session_state.get("selected_models", []) if name in model_names]
    selected_datasets = st.multiselect(
        "Datasets",
        options=dataset_names,
        default=previous_datasets or default_datasets,
        key="selected_datasets",
    )
    selected_models = st.multiselect(
        "Models",
        options=model_names,
        default=previous_models or default_models,
        key="selected_models",
    )

    if not selected_datasets or not selected_models:
        st.info("Select at least one dataset and one model to preview compatibility and build jobs.")
        st.session_state["planned_jobs"] = []
        return

    loaded_selected_pairs = set(loaded_pair_params)
    rows = []
    for ds_name in selected_datasets:
        for mod_name in selected_models:
            compat = (
                matrix_df.loc[ds_name, mod_name]
                if ds_name in matrix_df.index and mod_name in matrix_df.columns
                else "Incompatible"
            )
            default_selected = (
                (ds_name, mod_name) in loaded_selected_pairs
                if loaded_selected_pairs
                else compat in ("Compatible", "Partial")
            )
            rows.append({
                "Selected": default_selected and compat in ("Compatible", "Partial"),
                "Dataset": ds_name,
                "Model": mod_name,
                "Compatibility": compat,
            })
    editor_df = pd.DataFrame(rows)

    editor_signature = tuple((row["Dataset"], row["Model"], row["Compatibility"]) for row in rows)
    if st.session_state.get("job_matrix_signature") != editor_signature:
        st.session_state["job_matrix_signature"] = editor_signature
        bump_editor_version()
    editor_key = f"job_matrix_editor_v{int(st.session_state.get("editor_version", 0) or 0)}"

    edited = st.data_editor(
        editor_df,
        column_config={
            "Selected": st.column_config.CheckboxColumn("Selected", default=False),
            "Dataset": st.column_config.TextColumn("Dataset", disabled=True, width="medium"),
            "Model": st.column_config.TextColumn("Model", disabled=True, width="medium"),
            "Compatibility": st.column_config.TextColumn("Compatibility", disabled=True, width="small"),
        },
        width="stretch",
        hide_index=True,
        key=editor_key,
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
        return

    st.divider()
    st.subheader("2. Hyperparameter Overrides")
    model_name_to_schema_path = {m["name"]: m.get("hyperparameters_schema") for m in models}
    st.caption("Fields are generated from each model's hyperparameter schema.")
    pair_params: dict[tuple[str, str], dict] = {}

    for job in planned_jobs:
        ds_name = job["Dataset"]
        mod_name = job["Model"]
        job_key = f"{ds_name}::{mod_name}"
        field_key = f"params::{job_key}"
        expected_path = model_name_to_schema_path.get(mod_name) or f"schemas/models/{slugify_experiment_name(mod_name)}.hyperparameters.schema.json"

        with st.expander(f"{ds_name} × {mod_name}", expanded=False):
            schema = _load_hyperparameter_schema(model_name_to_schema_path.get(mod_name))
            loaded_params = _consume_loaded_pair_params(ds_name, mod_name)
            if loaded_params:
                _prefill_hyperparameter_widget_state(job_key, loaded_params)
            if schema and isinstance(schema.get("properties"), dict):
                pair_params[(ds_name, mod_name)] = render_hyperparameters_form(schema, job_key)
            else:
                st.info(
                    f"No hyperparameter schema found at `{expected_path}`. "
                    "Add a `hyperparameters.json` or schema path in `model.yaml` to enable form fields."
                )
                raw_default = json.dumps(loaded_params, indent=2, sort_keys=True) if loaded_params else "{}"
                if loaded_params:
                    st.session_state[field_key] = raw_default
                raw_params = st.text_area(
                    "Model Params (JSON)",
                    value=raw_default,
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

    st.session_state["pair_params"] = pair_params

    if st.button("Generate & Save Manifest", key="btn_gen_manifest"):
        try:
            manifest = build_run_manifest(
                experiment_name=experiment_name_input or "benchmark_run",
                random_seed=random_seed,
                run_mode=run_mode,
                planned_jobs=planned_jobs,
                dataset_name_to_slug=dataset_name_to_slug,
                pair_params=pair_params,
            )
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
            return

        manifest_path = manifest_path.strip() or "run_manifest.yaml"
        with open(manifest_path, "w") as fh:
            yaml.safe_dump(manifest, fh, default_flow_style=False, sort_keys=False)

        track("manifest_generated", n_jobs=len(planned_jobs), run_mode=run_mode, manifest_path=manifest_path)
        st.session_state["_pending_shared_manifest_path"] = manifest_path
        st.session_state["manifest_generated_path"] = manifest_path
        st.session_state["manifest_generated_yaml"] = yaml.safe_dump(
            manifest, default_flow_style=False, sort_keys=False
        )
        st.success(f"Manifest saved to `{manifest_path}`")

    generated_path = st.session_state.get("manifest_generated_path")
    if generated_path:
        st.code(f"make benchmark config={generated_path}")
        generated_yaml = st.session_state.get("manifest_generated_yaml")
        if generated_yaml:
            st.code(generated_yaml, language="yaml")
        if st.button("Proceed to Run →", key="btn_proceed_run"):
            go_to("run")


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
        width="stretch",
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


# ---------------------------------------------------------------------------
# Execute-tab subprocess streaming (S-04 fragment refactor)
# ---------------------------------------------------------------------------

_LOG_DEQUE_MAX = 1000
_LOG_DISPLAY_TAIL = 40


def _drain_pipe(pipe, out_q: "queue.Queue[str | None]") -> None:
    try:
        for line in iter(pipe.readline, ""):
            out_q.put(line)
    finally:
        out_q.put(None)
        try:
            pipe.close()
        except Exception:
            pass


def _launch_run_subprocess(
    *,
    manifest_file: Path,
    output_dir: str,
    seed: int,
    planned_jobs: list[dict],
    repo_root: Path,
) -> None:
    """Spawn the orchestrator subprocess and stash all live handles in
    session_state so the fragment monitor can poll them without rebuilding."""
    cmd = [
        sys.executable, "-m", "multiverse.runner.cli", "run",
        "--manifest", str(manifest_file),
        "--output", output_dir,
        "--seed", str(seed),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(repo_root),
    )
    stdout_q: "queue.Queue[str | None]" = queue.Queue()
    stderr_q: "queue.Queue[str | None]" = queue.Queue()
    threading.Thread(target=_drain_pipe, args=(proc.stdout, stdout_q), daemon=True).start()
    threading.Thread(target=_drain_pipe, args=(proc.stderr, stderr_q), daemon=True).start()

    st.session_state["_run_proc"] = proc
    st.session_state["_run_stdout_q"] = stdout_q
    st.session_state["_run_stderr_q"] = stderr_q
    st.session_state["_run_stdout_done"] = False
    st.session_state["_run_stderr_done"] = False
    st.session_state["_run_log_lines"] = deque(maxlen=_LOG_DEQUE_MAX)
    st.session_state["_run_job_statuses"] = {
        f"{j['Dataset']}_{j['Model']}": "Pending" for j in planned_jobs
    }
    st.session_state["is_running"] = True
    st.session_state["run_finalized"] = False
    st.session_state["_run_returncode"] = None


def _drain_run_queues() -> None:
    """Pull whatever lines are buffered from stdout/stderr without blocking."""
    log_lines: "deque[str]" = st.session_state["_run_log_lines"]
    job_statuses: dict[str, str] = st.session_state["_run_job_statuses"]
    stdout_q: "queue.Queue[str | None]" = st.session_state["_run_stdout_q"]
    stderr_q: "queue.Queue[str | None]" = st.session_state["_run_stderr_q"]

    while True:
        try:
            raw = stdout_q.get_nowait()
        except queue.Empty:
            break
        if raw is None:
            st.session_state["_run_stdout_done"] = True
            continue
        clean = _ANSI_RE.sub("", raw).rstrip()
        if clean:
            log_lines.append(clean)

    while True:
        try:
            raw = stderr_q.get_nowait()
        except queue.Empty:
            break
        if raw is None:
            st.session_state["_run_stderr_done"] = True
            continue
        clean = raw.strip()
        if not clean:
            continue
        try:
            event = json.loads(clean)
        except json.JSONDecodeError:
            log_lines.append(_ANSI_RE.sub("", clean))
            continue
        event_name = event.get("event")
        job_name = event.get("job")
        if event_name == "job_start" and job_name in job_statuses:
            job_statuses[job_name] = "Training"
        elif event_name == "job_end" and job_name in job_statuses:
            job_statuses[job_name] = "Done" if event.get("status") == "SUCCESS" else "Failed"
        elif event_name == "error":
            log_lines.append(f"ERROR {event.get('kind')}: {event.get('message', '')}")


def _finalize_run() -> None:
    """Mark stragglers as Done/Failed and emit the completion telemetry."""
    proc: subprocess.Popen = st.session_state["_run_proc"]
    job_statuses: dict[str, str] = st.session_state["_run_job_statuses"]
    rc = proc.returncode
    st.session_state["_run_returncode"] = rc
    if rc == 0:
        track("run_completed", status="success", returncode=rc)
        for k, v in job_statuses.items():
            if v == "Pending":
                job_statuses[k] = "Done"
    else:
        track("run_completed", status="failed", returncode=rc)
        for k, v in job_statuses.items():
            if v in ("Pending", "Training"):
                job_statuses[k] = "Failed"
    st.session_state["is_running"] = False
    st.session_state["run_finalized"] = True
    st.session_state["_run_just_finalized"] = True


@st.fragment(run_every=timedelta(seconds=1))
def _run_monitor_fragment() -> None:
    """Poll the subprocess + queues once per second.  All UI is owned by this
    fragment so the rest of the page does not rerun while the run is live."""
    proc: subprocess.Popen | None = st.session_state.get("_run_proc")
    if proc is None:
        return

    still_running = proc.poll() is None
    if still_running or not (
        st.session_state.get("_run_stdout_done") and st.session_state.get("_run_stderr_done")
    ):
        _drain_run_queues()

    if not still_running and not st.session_state.get("run_finalized"):
        _finalize_run()

    job_statuses: dict[str, str] = st.session_state["_run_job_statuses"]
    log_lines: "deque[str]" = st.session_state["_run_log_lines"]

    st.dataframe(
        pd.DataFrame([{"Job": k, "Status": v} for k, v in job_statuses.items()]),
        width="stretch",
        hide_index=True,
    )

    log_text = "\n".join(list(log_lines)[-_LOG_DISPLAY_TAIL:]) or "(no output yet)"
    components.html(
        "<pre id='run-log' "
        "style='height:400px;overflow-y:auto;white-space:pre-wrap;margin:0;"
        "font-family:monospace;font-size:0.875rem;'>"
        f"{html.escape(log_text)}"
        "</pre>"
        "<script>const el=document.getElementById('run-log');el.scrollTop=el.scrollHeight;</script>",
        height=420,
        scrolling=False,
    )

    if st.session_state.get("run_finalized") and st.session_state.get("_run_just_finalized"):
        st.session_state["_run_just_finalized"] = False
        st.rerun()


def _render_run_monitor() -> None:
    proc: subprocess.Popen | None = st.session_state.get("_run_proc")
    still_running = bool(proc and proc.poll() is None)
    if still_running:
        if not st.session_state.get("cancel_requested"):
            if st.button("Cancel Run", key="btn_cancel_run"):
                st.session_state["cancel_requested"] = True
                st.warning("Click again to confirm cancellation.")
        else:
            st.warning("Click again to confirm cancellation.")
            if st.button("Confirm Cancel Run", key="btn_confirm_cancel_run"):
                proc.terminate()
                st.session_state["cancel_requested"] = False
                st.warning("Pipeline cancellation requested.")

    with st.status("Pipeline log", expanded=True):
        _run_monitor_fragment()

    if st.session_state.get("run_finalized"):
        rc = st.session_state.get("_run_returncode")
        if rc == 0:
            st.success("Pipeline completed successfully.")
        else:
            st.error(f"Pipeline exited with code {rc}.")
        log_lines: "deque[str]" = st.session_state.get("_run_log_lines", deque())
        if log_lines:
            st.download_button(
                "Download pipeline log",
                data="\n".join(log_lines).encode(),
                file_name="pipeline_output.log",
                mime="text/plain",
                key="btn_download_pipeline_log",
            )


def _render_execute_tab() -> None:
    import psutil

    st.header("Run")

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
        st.dataframe(wave_df, width="stretch", hide_index=True)

    st.divider()

    # ------------------------------------------------------------------
    # T4.2: Live DAG Monitor
    # ------------------------------------------------------------------
    st.subheader("Launch & Monitor")

    manifest_path_input = st.text_input(
        "Run Manifest Path",
        value=st.session_state.get("shared_manifest_path", "run_manifest.yaml"),
        key="shared_manifest_path",
        help="Manifest generated in the Configure tab.",
    )
    output_dir_input = st.text_input(
        "Output Directory",
        value="store/artifacts/run_output",
        key="exec_output_dir",
    )
    exec_seed = st.number_input(
        "Random Seed", min_value=0, value=int(st.session_state.get("shared_seed", 42)), step=1, key="shared_seed"
    )

    if not planned_jobs:
        st.warning("No jobs planned. Go to the Configure tab first.")
        if st.button("Go to Configure →", key="shortcut_execute_jobs"):
            go_to("configure")

    if st.button(
        "Launch Run",
        key="btn_launch_run",
        disabled=(not planned_jobs or st.session_state.get("is_running", False)),
        help=None if planned_jobs else "Plan jobs in the Configure tab first.",
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest_file = Path(manifest_path_input.strip())
        if not manifest_file.is_absolute():
            manifest_file = (repo_root / manifest_file).resolve()
        if not manifest_file.exists():
            st.error(
                f"Manifest not found: `{manifest_file}`. "
                "Generate it in the Configure tab first."
            )
        else:
            from multiverse.runner.cli import parse_manifest

            conn = get_db_connection()
            try:
                parsed_manifest = parse_manifest(str(manifest_file), conn)
            finally:
                conn.close()
            if not parsed_manifest.ok:
                track("error_shown", component="execute_preflight", error_kind="manifest_validation")
                render_manifest_errors(parsed_manifest.errors)
                return

            track("run_launched", n_jobs=len(planned_jobs), manifest_path=str(manifest_file))
            _launch_run_subprocess(
                manifest_file=manifest_file,
                output_dir=output_dir_input.strip(),
                seed=int(exec_seed),
                planned_jobs=planned_jobs,
                repo_root=repo_root,
            )
            st.rerun()

    if st.session_state.get("is_running") or st.session_state.get("run_finalized"):
        _render_run_monitor()

    # ------------------------------------------------------------------
    # T4.3: Live MLflow Metrics
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Live MLflow Metrics")

    mlflow_base = _get_mlflow_url()
    if not _cached_service_status("mlflow", f"{mlflow_base}/health"):
        st.info("MLflow is offline — start it with `make services-up` to see live metrics.")
    else:
        exp_raw = st.session_state.get("shared_experiment_name", "benchmark_run") or "benchmark_run"
        try:
            exp_slug = slugify_experiment_name(exp_raw)
        except ValueError:
            exp_slug = "benchmark_run"
        st.caption(f"Monitoring MLflow experiment: `{exp_slug}` (edit in the Configure tab).")

        if exp_slug:
            _live_metrics_panel(exp_slug, mlflow_base)


# ---------------------------------------------------------------------------
# Tab: Results (stub)
# ---------------------------------------------------------------------------

def _count_runs(status_filter: str | None = None) -> int:
    """Count runs, optionally filtered by status."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (status_filter,))
        else:
            cursor.execute("SELECT COUNT(*) FROM runs")
        return int(cursor.fetchone()[0])
    finally:
        conn.close()


def _fetch_runs(
    status_filter: str | None = None,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """Query the runs table, optionally filtered by status and page."""
    init_db()
    conn = get_db_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        sql = (
            "SELECT r.run_id, r.dataset_id, d.name AS dataset_name, r.model_slug, r.model_version, "
            "r.model_name, r.status, r.output_path, r.failure_reason "
            "FROM runs r LEFT JOIN datasets d ON d.id = r.dataset_id"
        )
        params: list = []
        if status_filter:
            sql += " WHERE r.status = ?"
            params.append(status_filter)
        sql += " ORDER BY r.run_id DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
        cursor.execute(sql, params)
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


def _selected_run_from_summary(summary_df: pd.DataFrame, runs: list[dict]) -> dict:
    try:
        event = st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "Run ID": st.column_config.NumberColumn("Run ID", width="small"),
                "Dataset": st.column_config.TextColumn("Dataset", width="medium"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Output Path": st.column_config.TextColumn("Output Path", width="large"),
                "Failure Reason": st.column_config.TextColumn("Failure Reason", width="large"),
            },
        )
        selected_rows = getattr(getattr(event, "selection", None), "rows", []) or []
        selected_idx = int(selected_rows[0]) if selected_rows else 0
        return runs[selected_idx]
    except TypeError:
        st.dataframe(summary_df, width="stretch", hide_index=True)
        run_labels = [
            f"Run {r['run_id']} - {r['status']} - {r.get('dataset_name') or r.get('dataset_id') or 'unknown'} - {r['model_name'] or r.get('model_slug') or 'unknown'}"
            for r in runs
        ]
        selected_label = st.selectbox("Select a run", options=run_labels, key="results_run_selector")
        return runs[run_labels.index(selected_label)]


def _set_mlflow_context_from_job_spec(job_spec_file: Path, mlflow_base: str) -> str | None:
    if not job_spec_file.exists():
        return None
    try:
        spec_data = json.loads(job_spec_file.read_text(encoding="utf-8"))
        auto_exp_name = (
            spec_data.get("run_settings", {}).get("experiment_name")
            or spec_data.get("globals", {}).get("experiment_name")
            or spec_data.get("experiment_name")
        )
    except Exception:
        return None
    if not auto_exp_name:
        return None
    exp_id = _resolve_mlflow_experiment_id(auto_exp_name, mlflow_base)
    if exp_id:
        st.session_state["active_experiment_id"] = exp_id
        st.session_state["active_experiment_name"] = auto_exp_name
        return auto_exp_name
    return None


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
        if st.button("Refresh", key="btn_results_refresh"):
            st.rerun()

    filter_val = None if status_choice == "All" else status_choice
    runs = paginate(
        lambda: _count_runs(filter_val),
        lambda limit, offset: _fetch_runs(filter_val, limit=limit, offset=offset),
        page_size=50,
        key="results_page",
    )

    if not runs:
        st.info("No runs found. Launch a benchmarking run from the Run tab first.")
        if st.button("Go to Run →", key="shortcut_results_execute"):
            go_to("run")
        return

    summary_rows = [
        {
            "Run ID": r["run_id"],
            "Dataset": r.get("dataset_name") or r.get("dataset_id") or "-",
            "Model": r["model_name"] or r.get("model_slug") or "-",
            "Status": r["status"],
            "Output Path": r["output_path"] or "-",
            "Failure Reason": r.get("failure_reason") or "",
        }
        for r in runs
    ]
    summary_df = pd.DataFrame(summary_rows)
    selected_run = _selected_run_from_summary(summary_df, runs)

    st.subheader("Drill Down")
    st.write(
        f"Run {selected_run['run_id']} - "
        f"{selected_run.get('dataset_name') or selected_run.get('dataset_id') or 'unknown'} - "
        f"{selected_run.get('model_name') or selected_run.get('model_slug') or 'unknown'}"
    )

    if selected_run["status"] == "FAILED":
        track("error_shown", component="results_drilldown", error_kind="failed_run")
        st.error(selected_run.get("failure_reason") or "Run failed without a recorded failure reason.")

    if str(selected_run.get("failure_reason") or "").startswith("VALIDATION_ERROR:"):
        if st.button("Retry", key=f"retry_validation_{selected_run['run_id']}"):
            conn = get_db_connection()
            try:
                conn.execute(
                    "UPDATE runs SET status = 'QUEUED', failure_reason = NULL WHERE run_id = ?",
                    (selected_run["run_id"],),
                )
                conn.commit()
            finally:
                conn.close()
            st.rerun()

    artifact_dir = Path(selected_run["output_path"]) if selected_run["output_path"] else None
    job_spec_file = artifact_dir / "job_spec.json" if artifact_dir else None

    mlflow_base = _get_mlflow_url()
    mlflow_live = _cached_service_status("mlflow", f"{mlflow_base}/health")
    if mlflow_live and job_spec_file:
        exp_name = _set_mlflow_context_from_job_spec(job_spec_file, mlflow_base)
        if exp_name:
            if st.button("Open in Analysis →", key=f"btn_open_analysis_{selected_run['run_id']}"):
                go_to("analysis")
        else:
            with st.expander("Set MLflow experiment manually", expanded=False):
                manual_exp = st.text_input(
                    "MLflow experiment name",
                    value=st.session_state.get("active_experiment_name", ""),
                    key="manual_exp_name",
                    placeholder="benchmark_run",
                )
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
    elif not mlflow_live:
        st.caption("MLflow is offline; start it with `make services-up` to enable deep-linking.")

    if not artifact_dir or not artifact_dir.exists():
        st.warning(f"Artifact directory not found: `{artifact_dir}`")
        return

    from multiverse.tracking import find_metrics_json

    metrics_path = find_metrics_json(str(artifact_dir))
    if metrics_path:
        metrics_file = Path(metrics_path)
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
                    st.dataframe(metrics_df, width="stretch", hide_index=True)
                with col_chart:
                    chart_df = metrics_df.set_index("Metric")["Value"]
                    st.bar_chart(chart_df)
            else:
                st.warning("No numeric metrics available for this run.")
                st.json(raw_metrics)
        except Exception as exc:
            st.error(f"Could not parse metrics.json: {exc}")
    else:
        st.info("No metrics.json found in artifact directory.")

    st.subheader("Artifacts")
    render_artifact_tree(artifact_dir)

    log_file = artifact_dir / "container.log"
    with st.expander("Container Log", expanded=False):
        render_log_viewer(log_file)

    st.subheader("Provenance")
    st.caption(f"Artifact directory: `{artifact_dir}`")
    if job_spec_file and job_spec_file.exists():
        st.caption(f"Job spec: `{job_spec_file}`")
        render_download_button(job_spec_file, "Download job_spec.json")
        with st.expander("Job Spec", expanded=False):
            try:
                st.json(json.loads(job_spec_file.read_text(encoding="utf-8")))
            except Exception:
                st.text(job_spec_file.read_text(encoding="utf-8", errors="replace"))


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


def _cached_service_status(name: str, url: str, *, ttl: float = 10.0) -> bool:
    cache = st.session_state.setdefault("service_health_cache", {})
    now = time.monotonic()
    entry = cache.get(name)
    if entry and now - float(entry.get("checked_at", 0.0)) < ttl:
        return bool(entry.get("ok", False))
    ok = _check_service(url)
    cache[name] = {"ok": ok, "checked_at": now, "url": url}
    return ok


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
    mlflow_up = _cached_service_status("mlflow", f"{mlflow_base}/health")
    optuna_up = _cached_service_status("optuna", optuna_base)

    with st.sidebar:
        st.subheader("Services")
        mf_status = "● MLflow online" if mlflow_up else "○ MLflow offline"
        op_status = "● Optuna online" if optuna_up else "○ Optuna offline"
        st.write(mf_status)
        st.link_button("Open MLflow", mlflow_base, width="stretch")
        st.write(op_status)
        st.link_button("Open Optuna", f"{optuna_base}/dashboard", width="stretch")
        if not mlflow_up or not optuna_up:
            st.caption("Start services with `make services-up`.")

        st.divider()
        with st.expander("Settings", expanded=False):
            _render_settings_panel()


# ---------------------------------------------------------------------------
# Tab: Experiment Analysis (MLflow)
# ---------------------------------------------------------------------------

def _render_mlflow_tab() -> None:
    mlflow_base = _get_mlflow_url()
    mlflow_up = _cached_service_status("mlflow", f"{mlflow_base}/health")

    st.header("Analysis")

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
            if st.button("Clear filter", key="btn_mlflow_show_all", type="secondary"):
                st.session_state["active_experiment_id"] = None
                st.session_state["active_experiment_name"] = ""
                st.rerun()
    else:
        deep_url = mlflow_base
        st.caption(
            "No active experiment selected. "
            "Pick a run in the Results tab to deep-link here automatically."
        )
        if st.button("Go to Results →", key="shortcut_mlflow_results"):
            go_to("results")

    st.link_button("Open in new tab", deep_url)
    components.html(
        "<script>"
        "try {"
        "if (window.parent.location.protocol === 'https:') "
        "document.body.innerHTML = '<div style=\"font:14px sans-serif;color:#9a6700\">The embedded MLflow frame may be blocked because it is served over HTTP inside an HTTPS page.</div>';"
        "} catch(e) {}"
        "</script>",
        height=36,
    )
    components.iframe(deep_url, height=900, scrolling=True)


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

def _render_settings_panel() -> None:
    """Render Docker data-root controls inside the sidebar."""
    current_root = get_docker_data_root()

    st.caption(f"Default: `{DEFAULT_DOCKER_DATA_ROOT}`")
    st.write(f"Current: `{current_root}`")

    if os.path.exists(current_root):
        try:
            usage = shutil.disk_usage(current_root)
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            st.caption(f"{used_gb:.1f} GB used / {total_gb:.1f} GB total; {free_gb:.1f} GB free")
        except OSError as exc:
            st.warning(f"Could not read disk usage: {exc}")
    else:
        st.info("Path does not exist yet; Docker will create it on first use.")

    new_root = st.text_input(
        "Docker data root path",
        value=current_root,
        help="Absolute path where Docker will store images and containers.",
        key="settings_docker_data_root",
    )

    if st.button("Save configuration", key="btn_save_docker_root"):
        new_root = new_root.strip()
        if not new_root:
            st.error("Path must not be empty.")
        elif not os.path.isabs(new_root):
            st.error("Path must be absolute (start with '/').")
        else:
            st.session_state["pending_docker_data_root"] = new_root
            st.warning("Saving will restart the Docker daemon. Running benchmark jobs will be interrupted.")

    pending_root = st.session_state.get("pending_docker_data_root")
    if pending_root:
        st.warning("Confirm to restart Docker and apply the new data root.")
        if st.button("Save and restart Docker", key="btn_confirm_docker_restart"):
            cfg = get_config()
            cfg["docker_data_root"] = pending_root
            save_config(cfg)

            daemon_json_path = Path.home() / ".config" / "docker" / "daemon.json"
            daemon_json_path.parent.mkdir(parents=True, exist_ok=True)
            if daemon_json_path.exists():
                try:
                    with open(daemon_json_path) as fh:
                        daemon_cfg = json.load(fh)
                except json.JSONDecodeError:
                    daemon_cfg = {}
            else:
                daemon_cfg = {}

            daemon_cfg["data-root"] = pending_root
            with open(daemon_json_path, "w") as fh:
                json.dump(daemon_cfg, fh, indent=2)

            try:
                subprocess.run(
                    ["systemctl", "--user", "restart", "docker"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                st.error(f"Failed to restart Docker daemon: {exc.stderr.decode().strip()}")
                return

            deadline = time.monotonic() + 10.0
            ok = False
            while time.monotonic() < deadline:
                result = subprocess.run(
                    ["docker", "info"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if result.returncode == 0:
                    ok = True
                    break
                time.sleep(0.5)

            st.session_state.pop("pending_docker_data_root", None)
            if ok:
                st.success(f"Docker data root updated to `{pending_root}` and daemon restarted.")
            else:
                st.warning(
                    "Config saved and daemon restarted, but Docker did not become responsive within 10 s."
                )


def _render_settings_tab() -> None:
    st.header("Settings")
    _render_settings_panel()


def main() -> None:
    st.set_page_config(page_title="Multiverse", layout="wide")
    init_state()
    _render_observability_sidebar()

    st.title("Multiverse Benchmarking Platform")

    previous_tab = st.session_state.get("_last_tab")
    active_tab = render_top_nav()
    if previous_tab != active_tab:
        track("tab_switched", tab=active_tab, from_tab=previous_tab)
        st.session_state["_last_tab"] = active_tab

    if active_tab == "registry":
        _render_registry_tab()
    elif active_tab == "configure":
        _render_configure_tab()
    elif active_tab == "run":
        _render_execute_tab()
    elif active_tab == "results":
        _render_results_tab()
    elif active_tab == "analysis":
        _render_mlflow_tab()


if __name__ == "__main__":
    main()
