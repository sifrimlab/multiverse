# Developer Guide

This document provides technical details for developers who want to contribute to the Multi-verse project.

## How to Add a New Model

To integrate a new multimodal integration method, follow these steps:

### 1. Create the Model Wrapper
Create a new Python file in `multiverse/models/` (e.g., `new_model.py`). Define a class that inherits from `ModelFactory`.

```python
from .base import ModelFactory

class NewModel(ModelFactory):
    def __init__(self, dataset, dataset_name, config_path: str, is_gridsearch=False):
        super().__init__(dataset, dataset_name, model_name="new_model", config_path=config_path, is_gridsearch=is_gridsearch)
        # Initialize model-specific parameters from self.model_params.get("new_model")

    def train(self):
        # Implementation of the training process
        pass

    def evaluate_model(self):
        # Implementation of model-specific evaluation
        pass
```

### 2. Update the Model Registry
Add an entry for your model in `model_registry.json` at the project root. This allows the system to perform dynamic routing.

```json
{
  "name": "new_model",
  "docker_image": "multiverse-new_model:latest",
  "supported_omics": ["rna", "atac"]
}
```

### 3. Create a Dockerfile
Create a corresponding Dockerfile in the `containers/` directory. Ensure it installs all necessary dependencies and sets the entrypoint to run your model script.

### 4. Register in the Workflow
Add your model class to the `run_models_with_user_params` function in `multiverse/main.py`.

## Master-Worker Execution Flow

Multi-verse uses a Master-Worker architecture to ensure isolation and scalability.

### Master Process (`multiverse.runner.cli`)
1. **Validation**: Validates the user configuration against the Pydantic schema.
2. **Registry Lookup**: Loads `model_registry.json` to identify available models and their requirements.
3. **Omics Detection**: Inspects the input dataset to determine available omics (e.g., RNA, ATAC, ADT).
4. **Dynamic Routing**: Filters the user's requested models. A model is only executed if all its `supported_omics` are present in the dataset.
5. **Concurrent Image Preparation**: Pulls or builds the required Docker images in parallel using `asyncio`.
6. **Parallel Execution**: Spins up a Docker container for each eligible model.
   - **Isolation**: Each worker runs in its own container.
   - **Read-Only Mounts**: The input data directory is mounted as read-only (`ro`) to prevent accidental data corruption.
   - **Shared Output**: Each worker writes its results to a dedicated subdirectory in the output directory.

### Worker Process (Docker Containers)
Each worker container runs a specific model wrapper. It:
1. Loads the preprocessed data from `/data/input`.
2. Executes the `train()` method.
3. Saves latent embeddings to `/data/outputs/embeddings.h5`.
4. Generates a UMAP visualization at `/data/outputs/umap.png`.
5. Calculates model-specific metrics and saves them to `/data/outputs/metrics.json`.

### Result Aggregation
After all worker containers exit, the Master process:
1. Checks the exit codes of all containers.
2. Aggregates the `metrics.json` from successful runs.
3. Triggers the `Evaluator` to compute comparative `scIB-metrics` (ARI, NMI, etc.) across all successful models.
4. Generates a final `results.json` in the root output directory.
