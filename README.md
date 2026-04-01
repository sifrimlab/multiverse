# Multi-verse

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Multi-verse is a production-grade framework for the comparative analysis of multimodal single-cell data integration methods. It supports a variety of state-of-the-art models, including MOFA+, MOWGLI, MultiVI, and Cobolt, providing standardized evaluation metrics and visualizations.

<p align="center">
  <img src="logo.png" alt="Multi-verse Logo" width="200">
</p>

## Overview

Multi-verse simplifies the benchmarking of multimodal integration by providing:
- **Dynamic Routing**: Automatically filters eligible models based on the omics present in your dataset (RNA, ATAC, ADT).
- **Concurrent Orchestration**: Executes multiple models in parallel using isolated Docker containers for maximum performance and stability.
- **Standardized Evaluation**: Integrates `scIB-metrics` to calculate bio-conservation and batch-correction scores (ARI, NMI, Silhouette, etc.).
- **Interactive Setup**: A Streamlit-based wizard to generate configuration files without manual JSON editing.

---

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker (for containerized execution)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sifrimlab/multi-verse.git
   cd multi-verse
   ```

2. Install dependencies:
   ```bash
   make install
   ```

### Running the Setup Wizard
Launch the interactive GUI to generate your configuration:
```bash
make setup
```

### Running the Pipeline
You can run the pipeline directly or via the Docker orchestrator.

**Option 1: Direct Execution**
```bash
python -m multiverse.main config.json
```

**Option 2: Docker Orchestrator (Concurrent)**
```bash
python -m multiverse.runner.cli --concurrent --input /path/to/data --output /path/to/results --models pca mofa multivi
```

---

## Configuration Reference

The system uses a JSON configuration file. Below are the primary keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `batch_key` | `string` | **Required.** The key in `.obs` identifying experimental batches. |
| `cell_type_key` | `string` | Optional. The key in `.obs` identifying ground-truth cell types. |
| `random_seed` | `int` | Seed for reproducibility (default: 42). |
| `output_dir` | `string` | Path where results and logs will be saved. |
| `data` | `object` | Mapping of dataset names to their file paths and modality info. |
| `model` | `object` | Dictionary of models to run and their specific hyperparameters. |
| `_run_user_params` | `bool` | Whether to run models with the specified parameters. |

### Example `data` entry:
```json
"dataset_1": {
  "data_path": "data/pbmc.h5mu",
  "rna": { "file_name": "rna.h5ad", "is_preprocessed": false }
}
```

---

## Developer Guide

### Adding a New Model
1. **Implement the Wrapper**: Create a new class in `multiverse/models/` inheriting from `ModelFactory`.
2. **Update Registry**: Add your model metadata to `model_registry.json`.
   ```json
   {
     "name": "new_model",
     "docker_image": "multiverse-new_model:latest",
     "supported_omics": ["rna", "atac"]
   }
   ```
3. **Containerize**: Create a Dockerfile for your model and ensure it's built before running the orchestrator.

### Master-Worker Flow
1. **Master Process**: Validates config, loads the model registry, and checks dataset omics.
2. **Dynamic Routing**: Filters out models incompatible with the dataset's modalities.
3. **Preparation**: Pulls or builds required Docker images concurrently.
4. **Execution**: Spins up worker containers. Input data is mounted as **Read-Only**.
5. **Aggregation**: Once all workers finish, the master collects results and generates the final evaluation summary.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
