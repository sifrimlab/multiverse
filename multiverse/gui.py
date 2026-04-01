import json
import re

import pandas as pd
import streamlit as st
import yaml
from multiverse.registry import generate_compatibility_matrix
from multiverse.registry_db import get_all_datasets, get_all_models, init_db

@st.cache_data
def fetch_registry_data():
    """Fetches datasets and models from the SQLite registry."""
    # Ensure DB is initialized (idempotent)
    init_db()
    datasets = get_all_datasets()
    models = get_all_models()
    return datasets, models


def slugify_experiment_name(raw_name: str) -> str:
    """Convert experiment name into a filesystem-safe slug."""
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
    """Build the run manifest with top-level globals and jobs."""
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

def main():
    """Main entry point for the Streamlit-based setup wizard.

    Provides a graphical interface for users to specify their dataset details,
    select models, and generate a compatible YAML run manifest.
    """
    st.set_page_config(page_title="Multiverse Setup Wizard", layout="wide")
    st.title("Multiverse Setup Wizard")
    st.markdown("Use this interface to explore compatibility and generate your execution plan.")

    st.sidebar.header("Global Settings")
    experiment_name_input = st.sidebar.text_input(
        "Experiment Name",
        value="benchmark_run",
        help="Used to create store/artifacts/<experiment_name>/; it will be slugified.",
    )
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        step=1,
        value=42,
        help="Global random seed for reproducibility.",
    )
    run_mode = st.sidebar.radio(
        "Run Mode",
        options=["Use User Params", "Run Gridsearch"],
        index=0,
        help="Controls whether explicit params are used or model grid-search is enabled.",
    )

    datasets, models = fetch_registry_data()

    if not datasets:
        st.warning("No datasets found in registry. Please register a dataset first.")
        # Optional: Add a link or button to register datasets if T1.2 is implemented
        return

    st.subheader("1. Compatibility Matrix")
    matrix_df = generate_compatibility_matrix(datasets, models)

    # Styling the matrix
    def color_compatibility(val):
        color = 'black'
        if val == 'Compatible':
            color = '#90ee90' # Light green
        elif val == 'Partial':
            color = '#ffffe0' # Light yellow
        elif val == 'Incompatible':
            color = '#ffcccb' # Light red
        return f'background-color: {color}'

    st.dataframe(matrix_df.style.map(color_compatibility))

    st.subheader("2. Job Selection")

    col1, col2 = st.columns(2)

    dataset_name_to_slug = {
        d["name"]: (d.get("slug") or re.sub(r"[^a-zA-Z0-9._-]+", "-", d["name"]).strip("-").lower())
        for d in datasets
    }

    with col1:
        selected_dataset_names = st.multiselect(
            "Select Datasets",
            options=[d["name"] for d in datasets],
            help="Choose one or more datasets to process."
        )

    # Filter models based on selected datasets
    compatible_models_all = set()
    if selected_dataset_names:
        for ds_name in selected_dataset_names:
            compat_row = matrix_df.loc[ds_name]
            compatible_models = compat_row[compat_row.isin(["Compatible", "Partial"])].index.tolist()
            if not compatible_models_all:
                compatible_models_all = set(compatible_models)
            else:
                # We want models compatible with ALL selected datasets if they are to be multi-selected?
                # Actually, usually users want to run different models on different datasets.
                # But for a simple UI, let's show models compatible with AT LEAST ONE selected dataset.
                compatible_models_all.update(compatible_models)
    else:
        compatible_models_all = set([m["name"] for m in models])

    with col2:
        selected_model_names = st.multiselect(
            "Select Models",
            options=sorted(list(compatible_models_all)),
            help="Only models compatible with at least one selected dataset are shown."
        )

    st.subheader("3. Finalize Plan")

    # We need a way to pair them. Let's use a data editor for fine-grained control
    planned_jobs = []
    for ds_name in selected_dataset_names:
        for mod_name in selected_model_names:
            status = matrix_df.loc[ds_name, mod_name]
            if status in ["Compatible", "Partial"]:
                planned_jobs.append({"Dataset": ds_name, "Model": mod_name, "Status": status})
            else:
                # Optionally warn about incompatible pairs if they were somehow selected
                pass

    if planned_jobs:
        jobs_df = pd.DataFrame(planned_jobs)
        st.write("The following jobs will be added to the manifest:")
        st.table(jobs_df)

        st.markdown("#### Optional Model Hyperparameter Overrides")
        st.caption("Provide JSON per dataset-model pair (example: {\"factors\": 15}). Leave blank for {}.")
        pair_params = {}
        for job in planned_jobs:
            ds_name = job["Dataset"]
            mod_name = job["Model"]
            field_key = f"params::{ds_name}::{mod_name}"
            with st.expander(f"{ds_name} × {mod_name}", expanded=False):
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
                    pair_params[(ds_name, mod_name)] = None

        if st.button("Generate Run Manifest"):
            invalid_pairs = [
                (ds_name, model_name)
                for (ds_name, model_name), params in pair_params.items()
                if params is None
            ]
            if invalid_pairs:
                st.error("Fix invalid model parameter JSON fields before generating the manifest.")
                return

            try:
                manifest = build_run_manifest(
                    experiment_name=experiment_name_input,
                    random_seed=int(random_seed),
                    run_mode=run_mode,
                    planned_jobs=planned_jobs,
                    dataset_name_to_slug=dataset_name_to_slug,
                    pair_params=pair_params,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

            manifest_path = "run_manifest.yaml"
            with open(manifest_path, "w") as f:
                yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

            st.success(f"Manifest saved to {manifest_path}!")
            st.code(f"make benchmark config={manifest_path}")
            st.write("Manifest content:")
            st.code(yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False), language="yaml")
    else:
        st.info("Select datasets and compatible models to generate a plan.")

if __name__ == "__main__":
    main()
