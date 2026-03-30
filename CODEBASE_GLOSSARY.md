# Codebase Glossary – Multi-verse

## High-Level Overview

This project, "Multi-verse," is a Python-based package designed for the comparative analysis of various multimodal data integration methods, including MOFA, MOWGLI, MultiVI, and PCA. It processes single-cell genomics data, runs the specified models, and evaluates their performance using scIB metrics and UMAP visualizations. The project can be run either through a primary `main.py` script that uses a JSON configuration file to define datasets and model parameters, or through a command-line interface that launches Docker containers for each model. The core logic is organized within the `multiverse` package, which includes modules for data loading and preprocessing, model implementations, and runner scripts.

## Directory Tree

```
.
├── multiverse/    # Core package for the project
│   ├── runner/    # Scripts for running models (CLI and Docker)
│   ├── models/    # Implementations of the different models
│   ├── __init__.py
│   ├── main.py    # Main entry point for the pipeline
│   ├── config.py  # Configuration loading
│   ├── data_utils.py # Utility functions for data manipulation
│   ├── dataloader.py # Data loading and preprocessing
│   └── train.py   # Training and data preparation logic
├── docker-env/    # Dockerfiles for each model
├── docs/          # Documentation files
├── outputs/       # Directory for output files (e.g., plots, results)
├── .gitignore
├── config_alldatasets.json # Example configuration file
├── environment.yml # Conda environment definition
├── Makefile
└── README.md
```

## Glossary

### `multiverse/config.py`

---

**Name:** `load_config`

**Description:** Loads the configuration settings from a JSON file.

**Inputs:**
- `config_path` (str, optional): The path to the JSON configuration file. Defaults to `"./config.json"`.

**Outputs:**
- `dict`: A dictionary containing the hyperparameters and settings from the configuration file.

**Dependencies:**
- `json`: For parsing the JSON file.

**Dependents:**
- `multiverse/main.py`
- `multiverse/dataloader.py`
- `multiverse/models/base.py`
- `multiverse/train.py`
- All model files with a `main` function.

**Business Logic:**
1.  Opens the file at the specified `config_path`.
2.  Uses the `json` library to load the file content into a Python dictionary.
3.  Returns the dictionary.
4.  Raises a `RuntimeError` if the file cannot be loaded.

**Notes:**
- The function prints status messages to the console when it starts and successfully completes loading the file.

**Owner:**
- Not specified.

### `multiverse/data_utils.py`

---

**Name:** `fuse_mudata`

**Description:** Fuses a list of `AnnData` objects into a single `MuData` object, representing different modalities of a dataset. It ensures that the number of observations (cells) is the same across all modalities by intersecting them.

**Inputs:**
- `list_anndata` (List[ad.AnnData], optional): A list of `AnnData` objects to be fused.
- `list_modality` (List[str], optional): A list of strings representing the modality of each `AnnData` object.

**Outputs:**
- `md.MuData`: A `MuData` object containing the fused data.

**Dependencies:**
- `anndata` (as `ad`), `mudata` (as `md`), `muon` (as `mu`), `numpy` (as `np`)

**Dependents:**
- `multiverse/data_utils.py:anndata_concatenate`
- `multiverse/train.py:dataset_select`

**Business Logic:**
1.  Checks if the lengths of `list_anndata` and `list_modality` are equal.
2.  Creates a dictionary mapping modality names to `AnnData` objects.
3.  Creates a `MuData` object from the dictionary and intersects the observations.
4.  Handles cell type annotations by either using the one from the 'rna' modality or creating a default 'cell_type' column.

**Notes:**
- The function has a hard-coded logic to look for "cell_type" in the 'rna' modality's `.obs` attribute.

**Owner:**
- Not specified.

---

**Name:** `anndata_concatenate`

**Description:** Concatenates a list of `AnnData` objects along the variable axis to create a single `AnnData` object.

**Inputs:**
- `list_anndata` (List[ad.AnnData], optional): A list of `AnnData` objects to be concatenated.
- `list_modality` (List[str], optional): A list of strings representing the modality of each `AnnData` object.

**Outputs:**
- `ad.AnnData`: A single `AnnData` object containing the concatenated data.

**Dependencies:**
- `anndata` (as `ad`), `multiverse/data_utils.py:fuse_mudata`, `numpy` (as `np`)

**Dependents:**
- `multiverse/train.py:dataset_select`

**Business Logic:**
1.  Calls `fuse_mudata` to align the `AnnData` objects.
2.  Uses `ad.concat` to concatenate the objects along the variable axis.
3.  Adds 'cell_type' and 'modality' annotations.

**Owner:**
- Not specified.

### `multiverse/dataloader.py`

---

**Name:** `DataLoader` (Class)

**Description:** A class for loading single-cell data from various file formats into `AnnData` objects and triggering preprocessing.

**Methods:** `read_anndata`, `read_mudata`, `preprocessing`

**Owner:**
- Not specified.

---

**Name:** `Preprocessing` (Class)

**Description:** A class that handles the technical preprocessing of `AnnData` objects for different modalities (RNA, ATAC, ADT).

**Methods:** `rna_preprocessing`, `atac_preprocessing`, `adt_preprocessing`

**Owner:**
- Not specified.

### `multiverse/train.py`

---

**Name:** `load_datasets`

**Description:** Loads all datasets specified in the configuration file, using `DataLoader` to read and preprocess each modality.

**Inputs:**
- `config_path` (str): The path to the JSON configuration file.

**Outputs:**
- `dict`: A dictionary of datasets, where each dataset contains its modalities and a list of `AnnData` objects.

**Dependencies:**
- `multiverse.dataloader.DataLoader`, `multiverse.config.load_config`, `os`

**Dependents:**
- `multiverse/main.py` (via missing `Trainer` class), and all model `main` functions.

**Owner:**
- Not specified.

---

**Name:** `dataset_select`

**Description:** Prepares datasets for models by either concatenating modalities into a single `AnnData` object or fusing them into a `MuData` object.

**Inputs:**
- `datasets_dict` (dict): A dictionary of datasets from `load_datasets`.
- `data_type` (str): The desired output format ("concatenate" or "mudata").

**Outputs:**
- `dict`: A dictionary of datasets in the specified format.

**Dependencies:**
- `multiverse.data_utils.anndata_concatenate`, `multiverse.data_utils.fuse_mudata`

**Dependents:**
- All model `main` functions.

**Owner:**
- Not specified.

### `multiverse/runner/cli.py`

---

**Name:** `main`

**Description:** Provides a command-line interface for running selected models in Docker containers.

**Inputs:**
- Command-line arguments: `--models`, `--input`, `--output`.

**Outputs:**
- Triggers the execution of Docker containers.

**Dependencies:**
- `argparse`, `os`, `multiverse.runner.docker_runner.run_model_container`

**Owner:**
- Not specified.

### `multiverse/runner/docker_runner.py`

---

**Name:** `run_model_container`

**Description:** Runs a specified model within a Docker container, handling volume mapping and GPU support.

**Inputs:**
- `model_name` (str), `input_dir` (str), `output_dir` (str), `extra_args` (list, optional), `use_gpu` (bool, default=True)

**Outputs:**
- Starts a detached Docker container and streams its logs.

**Dependencies:**
- `docker`, `os`

**Dependents:**
- `multiverse/runner/cli.py`

**Owner:**
- Not specified.

### `multiverse/models/base.py`

---

**Name:** `ModelFactory` (Class)

**Description:** A base class that provides a common structure and interface for all models in the project.

**Methods:** `update_parameters`, `to`, `train`, `save_latent`, `load_latent`, `umap`, `evaluate_model`

**Notes:**
- This class acts as an abstract base class, ensuring that all models adhere to the same lifecycle. Subclasses are expected to override the placeholder methods.

**Owner:**
- Not specified.

### `multiverse/models/` (Model Implementations)

This section describes the specific model wrappers, which all inherit from `ModelFactory`.

---

**Common Pattern: `main` function in model files**

Each model file (`pca.py`, `mofa.py`, etc.) contains a `main` function that allows it to be run as a standalone script. This function typically performs the following steps:
1.  Parses a `--config_path` command-line argument.
2.  Loads datasets using `multiverse.train.load_datasets`.
3.  Prepares the data format using `multiverse.train.dataset_select` (`concatenate` for PCA/MultiVI, `mudata` for MOFA/Mowgli).
4.  Iterates through the datasets, instantiates the corresponding model class.
5.  Runs the model's pipeline (`to`, `train`, `save_latent`, `umap`, `evaluate_model`).
6.  Writes a log file.

---

**Name:** `PCA_Model` (Class in `pca.py`)

**Description:** A wrapper for Principal Component Analysis (PCA) using `scanpy`. It overrides the `train` method to perform PCA and the `evaluate_model` method to report the total explained variance.

**Owner:**
- Not specified.

---

**Name:** `MOFA_Model` (Class in `mofa.py`)

**Description:** A wrapper for the MOFA+ model using the `muon` library. It overrides `train` to run MOFA+ and `evaluate_model` to report the total explained variance.

**Owner:**
- Not specified.

---

**Name:** `Mowgli_Model` (Class in `mowgli.py`)

**Description:** A wrapper for the Mowgli model. It overrides `train` to run the Mowgli algorithm and `evaluate_model` to report the final Optimal Transport loss.

**Owner:**
- Not specified.

---

**Name:** `MultiVI_Model` (Class in `multivi.py`)

**Description:** A wrapper for the `scvi-tools` MultiVI model. It overrides `train` to run the MultiVI algorithm and `evaluate_model` to report the silhouette score of the latent embeddings.

**Owner:**
- Not specified.

---

**Name:** `Cobolt_Wrapper` (Class in `cobolt.py`)

**Description:** A wrapper for the Cobolt model. It overrides `train` to run the Cobolt algorithm and `evaluate_model` to report the final training loss.

**Owner:**
- Not specified.

### `multiverse/main.py`

---

**Name:** `main`

**Description:** The main entry point for running the Multi-verse pipeline in a local Python environment.

**Inputs:**
- `sys.argv[1]` (str): The path to the JSON configuration file.

**Outputs:**
- Writes output files to the directory specified in the configuration.

**Dependencies:**
- `sys`, `torch`, `multiverse.config.load_config`

**Business Logic:**
1.  Loads the configuration file.
2.  If `_run_user_params` is `True`, it runs the training and evaluation pipeline.
3.  If `_run_gridsearch` is `True`, it runs a grid search.

**Notes:**
- This file has several imports for modules that appear to be missing from the repository (`multiverse.train.Trainer`, `eval.Evaluator`, `utils.GridSearchRun`). This suggests the provided code is incomplete.

**Owner:**
- Not specified.
