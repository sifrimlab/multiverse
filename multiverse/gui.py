import streamlit as st
import json
import yaml
import os
import pandas as pd
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

def main():
    """Main entry point for the Streamlit-based setup wizard.

    Provides a graphical interface for users to specify their dataset details,
    select models, and generate a compatible YAML run manifest.
    """
    st.set_page_config(page_title="Multiverse Setup Wizard", layout="wide")
    st.title("Multiverse Setup Wizard")
    st.markdown("Use this interface to explore compatibility and generate your execution plan.")

    datasets, models = fetch_registry_data()

    if not datasets:
        st.warning("No datasets found in registry. Please register a dataset first.")
        # Optional: Add a link or button to register datasets if T1.2 is implemented
        return

    st.subheader("1. Compatibility Matrix")
    matrix_df = generate_compatibility_matrix(datasets, models)

    # Styling the matrix
    def color_compatibility(val):
        color = 'white'
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

        if st.button("Generate Run Manifest"):
            manifest = {
                "manifest_version": "1.0",
                "jobs": []
            }

            # Group by dataset for the manifest structure
            from collections import defaultdict
            ds_to_models = defaultdict(list)
            for job in planned_jobs:
                ds_to_models[job["Dataset"]].append(job["Model"])

            for ds_name, mods in ds_to_models.items():
                manifest["jobs"].append({
                    "dataset_id": ds_name,
                    "models": mods
                })

            manifest_path = "run_manifest.yaml"
            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False)

            st.success(f"Manifest saved to {manifest_path}!")
            st.code(f"make benchmark config={manifest_path}")
            st.write("Manifest content:")
            st.code(yaml.dump(manifest, default_flow_style=False), language="yaml")
    else:
        st.info("Select datasets and compatible models to generate a plan.")

if __name__ == "__main__":
    main()
