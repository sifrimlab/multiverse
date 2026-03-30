import streamlit as st
import json
import os
from multiverse.registry import load_registry

def main():
    st.title("Multiverse Setup Wizard")
    st.markdown("Use this interface to generate your system configuration file.")

    with st.form("setup_form"):
        # Basic Settings
        st.subheader("Basic Settings")
        dataset_path = st.text_input("Dataset Path (h5ad/h5mu)", value="data/dataset.h5mu")
        batch_key = st.text_input("Batch Key", value="batch")
        cell_type_key = st.text_input("Cell Type Key (Optional)", value="")
        output_dir = st.text_input("Output Directory", value="./outputs/")
        random_seed = st.number_input("Random Seed", value=42)

        # Model Selection
        st.subheader("Model Selection")
        try:
            registry = load_registry()
            model_options = list(registry.keys())
        except Exception as e:
            st.error(f"Failed to load model registry: {e}")
            model_options = []

        selected_models = st.multiselect("Select Models to Run", model_options, default=model_options[:1] if model_options else [])

        # Omics mapping (Simplified for GUI)
        st.subheader("Omics Configuration")
        rna_file = st.text_input("RNA Modality File Name (within dataset)", value="rna.h5ad")
        atac_file = st.text_input("ATAC Modality File Name (Optional)", value="")
        adt_file = st.text_input("ADT Modality File Name (Optional)", value="")

        submitted = st.form_submit_button("Generate Configuration")

        if submitted:
            # Construct the configuration dictionary matching SystemConfig schema
            config = {
                "batch_key": batch_key,
                "random_seed": int(random_seed),
                "output_dir": output_dir,
                "data": {
                    "dataset_1": {
                        "data_path": dataset_path,
                        "rna": {"file_name": rna_file}
                    }
                },
                "model": {m: {} for m in selected_models},
                "_run_user_params": True,
                "_run_gridsearch": False
            }

            if cell_type_key:
                 # In current schema, cell_type_key isn't in SystemConfig root,
                 # but it might be used by evaluation.
                 # For now we'll stick to the schema defined in T1.2 and T5.1
                 config["cell_type_key"] = cell_type_key

            if atac_file:
                config["data"]["dataset_1"]["atac"] = {"file_name": atac_file}
            if adt_file:
                config["data"]["dataset_1"]["adt"] = {"file_name": adt_file}

            config_filename = "generated_config.json"
            with open(config_filename, "w") as f:
                json.dump(config, f, indent=4)

            st.success(f"Configuration saved to {config_filename}!")
            st.json(config)

if __name__ == "__main__":
    main()
