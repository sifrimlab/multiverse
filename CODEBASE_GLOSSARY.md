# Codebase Glossary: Multi-verse

## High-Level Architecture Overview

Multi-verse is a modular framework designed for the comparative analysis of multimodal single-cell data integration methods. It follows a Master-Worker architecture where a central orchestrator (either `multiverse/main.py` for local execution or `multiverse/runner/cli.py` for Docker-based execution) manages the workflow. The system leverages `DataLoader` for standardized ingestion and preprocessing of omics data (RNA, ATAC, ADT). Models such as PCA, MOFA+, MultiVI, Mowgli, Cobolt, and TotalVI are implemented as subclasses of a common `ModelFactory`, ensuring a consistent interface for training, embedding generation, and visualization. Evaluation is performed using the `scIB-metrics` library to calculate bio-conservation and batch-correction scores. The technology stack includes Python, Docker for environment isolation, Pydantic for robust configuration validation, and Streamlit for an interactive setup wizard.

## Codebase Directory Tree

```
.
├── [containers/](#file-containers)        # Dockerfiles and environment specs for individual models
├── [docker-env/](#file-docker-env)        # Shared Docker environment configurations and requirement files
├── [docs/](#file-docs)              # Developer and user documentation
├── [multiverse/](#file-multiverse)        # Core package containing logic, models, and orchestration
│   ├── [models/](#file-multiverse-models)        # Model-specific wrappers inheriting from ModelFactory
│   └── [runner/](#file-multiverse-runner)        # Docker orchestration and CLI logic
├── [tests/](#file-tests)             # Unit and integration test suite
├── [Makefile](#file-makefile)            # Task runner for installation, setup, and execution
├── [config_alldatasets.json](#file-config-alldatasets-json) # Example configuration file
├── [model_registry.json](#file-model-registry-json) # Metadata for available models and their Docker images
├── [pyproject.toml](#file-pyproject-toml)      # Project dependencies and build configuration
├── [runner.py](#file-runner-py)           # Main entry point for local execution
└── [setup.py](#file-setup-py)            # Legacy setup script for package installation
```

## Glossary of Functions and Scripts

<a name="file-runner-py"></a>
### File: /runner.py

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Main entry point for the runner script providing a CLI for local workflow execution.
Inputs/Arguments     None (reads from `sys.argv`).
Outputs/Return Value None.
Dependencies         `multiverse.main.main_workflow`, `argparse`, `os`, `sys`.
Dependents (Used by) Executed as a standalone script.
Business Logic       Validates existence of the config file and invokes the main workflow.
Notes / Observations Defaults to `config_alldatasets.json` if no path is provided.
Owner/Team           Multi-verse Dev Team

---

<a name="file-setup-py"></a>
### File: /setup.py

Field                Description
-------------------  -----------------------------------------------------
Name                 setup() call
Description          Configures the `multiverse` package for installation.
Inputs/Arguments     Standard setuptools arguments (name, version, packages, etc.).
Outputs/Return Value None.
Dependencies         `setuptools`, `requirements.txt`.
Dependents (Used by) `pip install`, `pip install -e .`.
Business Logic       Defines package metadata and the `multiverse-cli` console script entry point.
Notes / Observations Reads requirements from `requirements.txt` dynamically.
Owner/Team           Multi-verse Dev Team

---

<a name="file-makefile"></a>
### File: /Makefile

Field                Description
-------------------  -----------------------------------------------------
Name                 install, setup, run, test, build-all, clean
Description          Phony targets for project management and execution.
Inputs/Arguments     Optional variables: `CONFIG_FILE`, `INPUT_DIR`, `OUTPUT_DIR`.
Outputs/Return Value Command execution results.
Dependencies         `uv`, `docker`, `python`, `pytest`.
Dependents (Used by) Developer CLI.
Business Logic       `install`: Syncs dependencies with `uv`. `setup`: Runs Streamlit GUI. `run`: Executes `runner.py`.
Notes / Observations Provides shortcuts for building model-specific Docker images.
Owner/Team           Multi-verse Dev Team

---

<a name="file-model-registry-json"></a>
### File: /model_registry.json

Field                Description
-------------------  -----------------------------------------------------
Name                 Registry Configuration
Description          A JSON file mapping model names to their Docker images and supported omics.
Inputs/Arguments     JSON object containing a list of `models`.
Outputs/Return Value None.
Dependencies         None.
Dependents (Used by) `multiverse.registry.load_registry`.
Business Logic       Defines the available model catalog and their capability requirements.
Notes / Observations Critical for the dynamic routing logic in `main_workflow`.
Owner/Team           Multi-verse Dev Team

---

<a name="file-multiverse"></a>
### File: /multiverse/main.py

Field                Description
-------------------  -----------------------------------------------------
Name                 set_seed(seed: int = 42)
Description          Sets random seeds for reproducibility across torch, numpy, and random.
Inputs/Arguments     seed (int): Random seed value. Defaults to 42.
Outputs/Return Value None.
Dependencies         `torch`, `numpy`, `random`.
Dependents (Used by) `main_workflow`.
Business Logic       Configures deterministic behavior for CUDA if available.
Notes / Observations Essential for ensuring consistent results across runs.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main_workflow(config_path: str)
Description          Orchestrates the end-to-end local execution of the integration pipeline.
Inputs/Arguments     config_path (str): Path to the JSON configuration file.
Outputs/Return Value None.
Dependencies         `load_config`, `validate_config`, `load_datasets`, `load_registry`, `get_eligible_models`, `setup_logging`, `run_models_with_user_params`.
Dependents (Used by) `/runner.py`.
Business Logic       Handles config validation, logging setup, dataset loading, and model routing based on available omics.
Notes / Observations Currently logs a warning that grid search is not yet implemented.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 run_models_with_user_params(config_path: str, datasets: dict, model_config: dict)
Description          Executes models sequentially using parameters from the config.
Inputs/Arguments     config_path (str), datasets (dict), model_config (dict).
Outputs/Return Value None.
Dependencies         `PCAModel`, `MOFAModel`, `MultiVIModel`, `MowgliModel`, `CoboltModel`, `TotalVIModel`, `dataset_select`.
Dependents (Used by) `main_workflow`.
Business Logic       Instantiates and runs the pipeline (train, save_latent, umap, evaluate) for each eligible model.
Notes / Observations Uses `dataset_select('concatenate')` to prepare data for models.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/config.py

Field                Description
-------------------  -----------------------------------------------------
Name                 load_config(config_path: str = "./config.json")
Description          Loads the configuration from a JSON file.
Inputs/Arguments     config_path (str): Path to the JSON file. Defaults to "./config.json".
Outputs/Return Value config (dict): Parsed configuration data.
Dependencies         `json`, `logging_utils`.
Dependents (Used by) `main_workflow`, `load_datasets`, `Evaluator`, `Preprocessing`, `ModelFactory`.
Business Logic       Reads and parses JSON, handling basic file-not-found and decoding errors.
Notes / Observations Root logger is used via `get_logger`.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/config_schema.py

Field                Description
-------------------  -----------------------------------------------------
Name                 validate_config(config_data: Dict[str, Any])
Description          Validates a raw configuration dictionary against the Pydantic SystemConfig schema.
Inputs/Arguments     config_data (Dict[str, Any]): Raw configuration.
Outputs/Return Value SystemConfig: Validated Pydantic object.
Dependencies         `Pydantic`, `SystemConfig`.
Dependents (Used by) `main_workflow`.
Business Logic       Enforces type checking and required fields (like `batch_key`).
Notes / Observations Uses `field_validator` to check if data paths exist on the host.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 validate_path_exists(cls, v)
Description          Pydantic validator to ensure a file path exists.
Inputs/Arguments     v (str): The path to validate.
Outputs/Return Value v (str): The validated path.
Dependencies         `os.path.exists`.
Dependents (Used by) `DatasetConfig`.
Business Logic       Raises `ValueError` if the path does not exist.
Notes / Observations Part of the `DatasetConfig` Pydantic model.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/data_utils.py

Field                Description
-------------------  -----------------------------------------------------
Name                 fuse_mudata(list_anndata: List[ad.AnnData], list_modality: List[str])
Description          Fuses a list of AnnData objects into a single MuData object.
Inputs/Arguments     list_anndata (List[ad.AnnData]), list_modality (List[str]).
Outputs/Return Value data (md.MuData): Fused MuData object.
Dependencies         `muon`, `numpy`.
Dependents (Used by) `anndata_concatenate`, `dataset_select`.
Business Logic       Uses `intersect_obs` to ensure observation consistency across modalities.
Notes / Observations Standardizes 'cell_type' annotation.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 anndata_concatenate(list_anndata: List[ad.AnnData], list_modality: List[str])
Description          Concatenates multiple AnnData objects along the variable axis.
Inputs/Arguments     list_anndata (List[ad.AnnData]), list_modality (List[str]).
Outputs/Return Value anndata (ad.AnnData): Concatenated object.
Dependencies         `anndata`, `fuse_mudata`.
Dependents (Used by) `dataset_select`, `CoboltModel.main`.
Business Logic       First fuses to MuData for alignment, then concatenates.
Notes / Observations Adds a `modality` observation column.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 load_datasets(config_path_or_dict: Union[str, dict])
Description          Loads and preprocesses all datasets specified in the configuration.
Inputs/Arguments     config_path_or_dict (Union[str, dict]): Config path or pre-loaded dict.
Outputs/Return Value datasets (dict): Map of dataset names to their modalities and AnnData objects.
Dependencies         `DataLoader`, `load_config`.
Dependents (Used by) `main_workflow`, `Evaluator`, model module main functions.
Business Logic       Iterates through config datasets and invokes `DataLoader` for each modality.
Notes / Observations Handles both path and pre-loaded dict for config.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 dataset_select(datasets_dict: dict, data_type: str)
Description          Converts internal dataset dictionary into model-ready formats ('concatenate' or 'mudata').
Inputs/Arguments     datasets_dict (dict), data_type (str): "concatenate" or "mudata".
Outputs/Return Value data (dict): Processed data objects.
Dependencies         `anndata_concatenate`, `fuse_mudata`.
Dependents (Used by) `main_workflow`, `Evaluator`, model module main functions.
Business Logic       Routes data formatting based on model requirements.
Notes / Observations PCA/MultiVI use 'concatenate'; MOFA/Mowgli use 'mudata'.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/dataloader.py

Field                Description
-------------------  -----------------------------------------------------
Name                 DataLoader.read_anndata()
Description          Reads various file formats into a standardized AnnData object.
Inputs/Arguments     None.
Outputs/Return Value data (ad.AnnData).
Dependencies         `scanpy`, `muon`, `anndata`.
Dependents (Used by) `DataLoader.preprocessing()`.
Business Logic       Supports .csv, .tsv, .h5ad, .txt, .mtx, .h5mu, .h5. Standardizes 'cell_type' key.
Notes / Observations Handles 10X mtx and h5 files via muon.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 DataLoader.read_mudata()
Description          Reads a file into a MuData object.
Inputs/Arguments     None.
Outputs/Return Value data (md.MuData).
Dependencies         `muon`, `mudata`.
Dependents (Used by) Potential future multi-modal ingestion.
Business Logic       Supports .h5mu, .h5, .mtx.
Notes / Observations Standardizes ingestion for MuData-native formats.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 DataLoader.preprocessing()
Description          Orchestrates loading and modality-specific technical preprocessing.
Inputs/Arguments     None.
Outputs/Return Value data (ad.AnnData).
Dependencies         `read_anndata`, `Preprocessing` class.
Dependents (Used by) `load_datasets`.
Business Logic       Triggers RNA, ATAC, or ADT preprocessing if data is not already processed.
Notes / Observations Copies raw counts to `layers["counts"]` before processing.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 Preprocessing.rna_preprocessing()
Description          Performs QC, normalization, and feature selection for RNA data.
Inputs/Arguments     None.
Outputs/Return Value data (ad.AnnData).
Dependencies         `scanpy`, `muon`.
Dependents (Used by) `DataLoader.preprocessing()`.
Business Logic       Filters by gene counts, total counts, and MT percentage. Uses Seurat flavor for HVGs.
Notes / Observations Parameters are fetched from the system config's `preprocess_params`.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 Preprocessing.atac_preprocessing()
Description          Performs QC, normalization, and feature selection for ATAC data.
Inputs/Arguments     None.
Outputs/Return Value data (ad.AnnData).
Dependencies         `scanpy`, `muon`.
Dependents (Used by) `DataLoader.preprocessing()`.
Business Logic       Filters peaks based on cell counts and performs per-cell normalization.
Notes / Observations Uses `n_top_peaks` for feature selection.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 Preprocessing.adt_preprocessing()
Description          Performs normalization for ADT (protein) data.
Inputs/Arguments     None.
Outputs/Return Value data (ad.AnnData).
Dependencies         `muon.prot`.
Dependents (Used by) `DataLoader.preprocessing()`.
Business Logic       Removes "total" feature and performs CLR normalization.
Notes / Observations Sets `feature_types` to "Protein Expression".
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/evaluate.py

Field                Description
-------------------  -----------------------------------------------------
Name                 determine_valid_metrics(config, dataset, requested_metrics)
Description          Filters metrics based on available dataset metadata (batch, cell type).
Inputs/Arguments     config (dict), dataset (ad.AnnData), requested_metrics (dict).
Outputs/Return Value valid_metrics (dict): List of metrics that can be safely computed.
Dependencies         None.
Dependents (Used by) Internal logic or future orchestrators.
Business Logic       Skips supervised metrics if labels are missing; skips batch metrics if only one batch exists.
Notes / Observations Prevents crashes during scIB-metrics execution.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 aggregate_results(model_status: dict, output_dir: str)
Description          Aggregates metrics from successful model runs into a single JSON file.
Inputs/Arguments     model_status (dict), output_dir (str).
Outputs/Return Value final_results (dict).
Dependencies         `json`, `os`.
Dependents (Used by) Integrated reporting logic.
Business Logic       Reads `metrics.json` from each model directory and compiles them.
Notes / Observations Saves to `results.json` at the output root.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 Evaluator.load_embeddings()
Description          Loads latent embeddings for each model from the output directory.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `h5py`, `os`.
Dependents (Used by) `Evaluator.main()`.
Business Logic       Reads `embeddings.h5` for each model and stores them in `self.dataset.obsm`.
Notes / Observations Key format: `X_<model_name>`.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 Evaluator.evaluate_models()
Description          Runs the scIB-metrics benchmark suite on all loaded model embeddings.
Inputs/Arguments     batch_key (str), label_key (str).
Outputs/Return Value metrics (dict).
Dependencies         `scib_metrics.benchmark.Benchmarker`.
Dependents (Used by) `main_workflow` (indirectly), `Evaluator.main()`.
Business Logic       Computes ARI, NMI, Silhouette, Graph Connectivity, etc. Generates a summary table plot.
Notes / Observations Handles missing batch/label keys by injecting dummies or skipping.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/gui.py

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Launches the Streamlit-based setup wizard.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `streamlit`, `load_registry`.
Dependents (Used by) Standalone script execution (`make setup`).
Business Logic       Collects user input via a form and generates `generated_config.json`.
Notes / Observations Simplifies complex JSON configuration for the user.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/registry.py

Field                Description
-------------------  -----------------------------------------------------
Name                 load_registry(registry_path: str = "model_registry.json")
Description          Loads model metadata from a JSON registry file.
Inputs/Arguments     registry_path (str). Defaults to "model_registry.json".
Outputs/Return Value registry (Dict[str, ModelEntry]).
Dependencies         `json`, Pydantic (`ModelRegistry`).
Dependents (Used by) `main_workflow`, `gui.py`, `run_workflow_async`.
Business Logic       Parses model names, Docker images, and supported omics.
Notes / Observations Searches for registry in current and parent directories.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 get_eligible_models(user_requested_models, available_omics, registry)
Description          Filters requested models based on omics compatibility with the dataset.
Inputs/Arguments     user_requested_models (List[str]), available_omics (List[str]), registry.
Outputs/Return Value eligible_models (List[str]).
Dependencies         None.
Dependents (Used by) `main_workflow`.
Business Logic       A model is eligible if its required omics are a subset of available omics.
Notes / Observations Logs warnings for missing omics.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/ingestion.py

Field                Description
-------------------  -----------------------------------------------------
Name                 load_dataset(file_path: str)
Description          Loads a single-cell dataset from a file (.h5ad or .h5mu).
Inputs/Arguments     file_path (str): The path to the dataset file.
Outputs/Return Value data (ad.AnnData or md.MuData).
Dependencies         `scanpy`, `muon`.
Dependents (Used by) `DataLoader`.
Business Logic       Validates file existence and format before reading.
Notes / Observations Raises `FileNotFoundError` or `ValueError` on failure.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 validate_dataset_structure(data, batch_key, cell_type_key)
Description          Verifies internal structural requirements of the dataset.
Inputs/Arguments     data (AnnData/MuData), batch_key (str), cell_type_key (str).
Outputs/Return Value omics (List[str]): List of available modalities.
Dependencies         `scanpy`, `mudata`.
Dependents (Used by) Internal dataset validation logic.
Business Logic       Ensures required keys exist in `.obs` and identifies omics.
Notes / Observations Critical for downstream routing and evaluation.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/logging_utils.py

Field                Description
-------------------  -----------------------------------------------------
Name                 setup_logging(log_dir: str, log_level: int)
Description          Configures the root logger to write to a file in the output directory.
Inputs/Arguments     log_dir (str), log_level (int). Defaults to `logging.INFO`.
Outputs/Return Value None.
Dependencies         `logging`, `os`.
Dependents (Used by) `main_workflow`, `run_workflow_async`, model main functions.
Business Logic       Initializes file handler and sets formatting. Clears existing handlers to avoid duplicates.
Notes / Observations Log file is named `multiverse.log`.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 get_logger(name: str)
Description          Returns a logger instance with the given name.
Inputs/Arguments     name (str).
Outputs/Return Value logger (logging.Logger).
Dependencies         `logging`.
Dependents (Used by) All modules in the `multiverse` package.
Business Logic       Standard wrapper for `logging.getLogger`.
Notes / Observations Ensures consistent logging across the package.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/utils.py

Field                Description
-------------------  -----------------------------------------------------
Name                 get_device(device_str: str)
Description          Creates a torch.device object from a string identifier.
Inputs/Arguments     device_str (str): e.g., "cpu", "cuda:0".
Outputs/Return Value device (torch.device).
Dependencies         `torch`.
Dependents (Used by) `MultiVIModel`, `TotalVIModel`, `CoboltModel`, `MowgliModel`.
Business Logic       Defaults to CPU if GPU is requested but unavailable.
Notes / Observations Logs availability information.
Owner/Team           Multi-verse Dev Team

---

<a name="file-multiverse-models"></a>
### File: /multiverse/models/base.py

Field                Description
-------------------  -----------------------------------------------------
Name                 ModelFactory.update_parameters(**kwargs)
Description          Updates the model attributes with new parameter values.
Inputs/Arguments     **kwargs: Dictionary of attribute names and new values.
Outputs/Return Value None.
Dependencies         None.
Dependents (Used by) Potential hyperparameter optimization loops.
Business Logic       Uses `hasattr` and `setattr` to safely update instance variables.
Notes / Observations Logs warnings for invalid parameter names.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 ModelFactory.save_latent()
Description          Saves the calculated latent representation of the data to an HDF5 file.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `h5py`, `numpy`.
Dependents (Used by) Model subclasses (`PCAModel`, `MOFAModel`, etc.).
Business Logic       Standardizes output format for embeddings across all models.
Notes / Observations Uses `self.latent_key` to fetch data from `self.dataset.obsm`.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 ModelFactory.umap()
Description          Generates and saves a UMAP visualization using the model's latent embeddings.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `scanpy`, `matplotlib`.
Dependents (Used by) Model subclasses.
Business Logic       Computes neighbors and UMAP based on model-specific latent space.
Notes / Observations Colors by `umap_color_type` if available in metadata.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/models/pca.py

Field                Description
-------------------  -----------------------------------------------------
Name                 PCAModel.train()
Description          Calculates principal components on the concatenated dataset.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `scanpy.pp.pca`.
Dependents (Used by) `run_models_with_user_params`.
Business Logic       Optionally uses highly variable genes for PCA computation.
Notes / Observations Extracts variance ratio for evaluation.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Standalone script for running the PCA model.
Inputs/Arguments     --config_path (str).
Outputs/Return Value None.
Dependencies         `PCAModel`, `load_datasets`, `dataset_select`, `setup_logging`.
Dependents (Used by) Docker container entry point.
Business Logic       Loads data, trains PCA, saves latent space, and generates UMAP.
Notes / Observations Entry point for `multiverse-pca` container.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/models/mofa.py

Field                Description
-------------------  -----------------------------------------------------
Name                 MOFAModel.train()
Description          Trains the MOFA+ model using variational inference via `muon`.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `muon.tl.mofa`.
Dependents (Used by) `run_models_with_user_params`.
Business Logic       Handles GPU/CPU routing and computes explained variance per factor.
Notes / Observations Uses MuData structure internally.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 MOFAModel._compute_explained_variance()
Description          Computes the variance explained by each latent factor.
Inputs/Arguments     None.
Outputs/Return Value explained_variance_ratio (np.ndarray).
Dependencies         `numpy`.
Dependents (Used by) `MOFAModel.train()`.
Business Logic       Aggregates total variance across all modalities and computes capture ratio.
Notes / Observations Used when `muon` does not provide explained variance automatically.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Standalone script for running the MOFA+ model.
Inputs/Arguments     --config_path (str).
Outputs/Return Value None.
Dependencies         `MOFAModel`, `load_datasets`, `dataset_select`, `setup_logging`.
Dependents (Used by) Docker container entry point.
Business Logic       Loads data as MuData, trains MOFA+, and evaluates explained variance.
Notes / Observations Entry point for `multiverse-mofa` container.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/models/multivi.py

Field                Description
-------------------  -----------------------------------------------------
Name                 MultiVIModel.train()
Description          Trains the MultiVI model using stochastic variational inference.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `scvi.model.MULTIVI`.
Dependents (Used by) `run_models_with_user_params`.
Business Logic       Requires `feature_types` in dataset to distinguish RNA and ATAC.
Notes / Observations Inherits from `ModelFactory`.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Standalone script for running the MultiVI model.
Inputs/Arguments     --config_path (str).
Outputs/Return Value None.
Dependencies         `MultiVIModel`, `load_datasets`, `dataset_select`, `setup_logging`.
Dependents (Used by) Docker container entry point.
Business Logic       Orchestrates MultiVI training and evaluation using scVI tools.
Notes / Observations Entry point for `multiverse-multivi` container.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/models/totalvi.py

Field                Description
-------------------  -----------------------------------------------------
Name                 TotalVIModel.train()
Description          Trains the TotalVI model using variational inference.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `scvi.model.TOTALVI`.
Dependents (Used by) `run_models_with_user_params`.
Business Logic       Handles joint analysis of RNA and protein data.
Notes / Observations Requires `protein_expression` in `.obsm`.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Standalone script for running the TotalVI model.
Inputs/Arguments     --config_path (str).
Outputs/Return Value None.
Dependencies         `TotalVIModel`, `load_datasets`, `dataset_select`, `setup_logging`.
Dependents (Used by) Docker container entry point.
Business Logic       Orchestrates TotalVI training on RNA+Protein datasets.
Notes / Observations Entry point for `multiverse-totalvi` container.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/models/cobolt.py

Field                Description
-------------------  -----------------------------------------------------
Name                 CoboltModel.train()
Description          Trains the Cobolt model using a Bayesian hierarchical framework.
Inputs/Arguments     None (uses `self.num_epochs`).
Outputs/Return Value None.
Dependencies         `cobolt.model.Cobolt`.
Dependents (Used by) `run_models_with_user_params`.
Business Logic       Initializes `MultiomicDataset` from multiple `SingleData` objects.
Notes / Observations Saves latent embeddings for the intersection of cells.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Standalone script for running the Cobolt model.
Inputs/Arguments     --config_path (str).
Outputs/Return Value None.
Dependencies         `CoboltModel`, `load_datasets`, `setup_logging`.
Dependents (Used by) Docker container entry point.
Business Logic       Iterates through datasets, training Cobolt on multiple modalities.
Notes / Observations Entry point for `multiverse-cobolt` container.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/models/mowgli.py

Field                Description
-------------------  -----------------------------------------------------
Name                 MowgliModel.train()
Description          Trains the Mowgli model using Optimal Transport and NMF.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `mowgli.models.MowgliModel`.
Dependents (Used by) `run_models_with_user_params`.
Business Logic       Configures optimizer, learning rate, and tolerance from parameters.
Notes / Observations Captures final OT loss for evaluation.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Standalone script for running the Mowgli model.
Inputs/Arguments     --config_path (str).
Outputs/Return Value None.
Dependencies         `MowgliModel`, `load_datasets`, `dataset_select`, `setup_logging`.
Dependents (Used by) Docker container entry point.
Business Logic       Trains Mowgli using Optimal Transport on MuData.
Notes / Observations Entry point for `multiverse-mowgli` container.
Owner/Team           Multi-verse Dev Team

---

<a name="file-multiverse-runner"></a>
### File: /multiverse/runner/docker_runner.py

Field                Description
-------------------  -----------------------------------------------------
Name                 build_images_concurrently(image_tags: list, status_callback: callable)
Description          Ensures all required Docker images for models are prepared concurrently.
Inputs/Arguments     image_tags (list), status_callback (callable).
Outputs/Return Value None.
Dependencies         `docker`, `asyncio`.
Dependents (Used by) `run_workflow_async`.
Business Logic       Uses an executor to run blocking Docker SDK calls concurrently.
Notes / Observations Updates status via callback for real-time dashboard.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 run_models_concurrently(models_info, data_path, seed, output_dir, status_callback)
Description          Executes eligible models in parallel using isolated Docker containers.
Inputs/Arguments     models_info (list), data_path (str), seed (int), output_dir (str), status_callback.
Outputs/Return Value summary (dict): Mapping of model names to "success" or "failed".
Dependencies         `docker`, `asyncio`, `os`.
Dependents (Used by) `run_workflow_async`.
Business Logic       Mounts data as read-only, captures logs on failure, and manages container lifecycle.
Notes / Observations Isolated execution prevents dependency conflicts between models.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 run_model_container(model_name: str, input_dir: str, output_dir: str, ...)
Description          Runs a single model container synchronously.
Inputs/Arguments     model_name (str), input_dir (str), output_dir (str), extra_args (list), use_gpu (bool).
Outputs/Return Value None.
Dependencies         `docker`.
Dependents (Used by) Legacy sequential execution in `cli.py`.
Business Logic       Maps model name to image and mounts volumes for input/output.
Notes / Observations Supports optional GPU device requests.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 run_evaluation_container(input_dir, output_dir, extra_args, use_gpu)
Description          Runs the evaluation metrics container synchronously.
Inputs/Arguments     input_dir (str), output_dir (str), extra_args (list), use_gpu (bool).
Outputs/Return Value None.
Dependencies         `docker`.
Dependents (Used by) `run_workflow_async`, `cli.main`.
Business Logic       Executes the `multiverse-evaluate` image on integrated data.
Notes / Observations Aggregates results from multiple models.
Owner/Team           Multi-verse Dev Team

---

### File: /multiverse/runner/cli.py

Field                Description
-------------------  -----------------------------------------------------
Name                 generate_status_table(tasks: Dict[str, str])
Description          Generates a Rich Table representing the current status of all tasks.
Inputs/Arguments     tasks (Dict[str, str]): Mapping from task names to status.
Outputs/Return Value table (rich.table.Table).
Dependencies         `rich.table.Table`.
Dependents (Used by) `run_workflow_async`.
Business Logic       Colors rows based on status (Success/Ready: Green, Failed/Error: Red, else Yellow).
Notes / Observations Formatted for terminal display.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 run_workflow_async(args: argparse.Namespace)
Description          Executes the concurrent Docker-based workflow with a live dashboard.
Inputs/Arguments     args (argparse.Namespace): Parsed CLI arguments.
Outputs/Return Value None.
Dependencies         `Live`, `Table` (from rich), `build_images_concurrently`, `run_models_concurrently`.
Dependents (Used by) `cli.main()`.
Business Logic       Orchestrates image prep, model execution, and final evaluation.
Notes / Observations Provides a visual progress dashboard in the terminal.
Owner/Team           Multi-verse Dev Team

Field                Description
-------------------  -----------------------------------------------------
Name                 main()
Description          Main entry point for the multiverse CLI.
Inputs/Arguments     None.
Outputs/Return Value None.
Dependencies         `argparse`, `run_workflow_async`, `run_model_container`.
Dependents (Used by) `multiverse-cli` console script.
Business Logic       Parses arguments and routes to either sequential or concurrent execution.
Notes / Observations Configures logging and output directories.
Owner/Team           Multi-verse Dev Team

---

<a name="file-tests"></a>
### File: /tests/simulate_dashboard.py

Field                Description
-------------------  -----------------------------------------------------
Name                 Standalone Script
Description          Mock script to demonstrate the Rich-based execution dashboard.
Inputs/Arguments     None.
Outputs/Return Value Terminal output.
Dependencies         `rich`.
Dependents (Used by) None.
Business Logic       Simulates task progression with random sleeps.
Notes / Observations Used for UI/UX testing of the CLI dashboard.
Owner/Team           Multi-verse Dev Team

---

<a name="file-containers"></a>
### Directory: /containers/
Description: Contains model-specific Dockerfiles and conda environment files. Each subdirectory (e.g., `pca`, `mofa`) defines an isolated execution environment for a specific integration method.

---

<a name="file-docker-env"></a>
### Directory: /docker-env/
Description: Shared environment resources, including a general Dockerfile for the evaluation container and various `requirements.txt` files for different model dependencies.

---

<a name="file-docs"></a>
### Directory: /docs/
Description: Project documentation, including a developer guide, model container specifications, and instructions for the runner.
