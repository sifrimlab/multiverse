# Epic [E1]: Foundation - Project Structure & Data Pipeline

Description
Establish the reproducible environment, centralized configuration system, and single-source-of-truth data pipeline. This prevents pipeline runs from failing due to environment discrepancies or malformed data.

Outcome
A robust baseline capable of deterministic dependency installation, strict configuration validation, and centralized data preprocessing.

---

## Sprint [S1]: Environment & Configuration

Goal
Set up dependency management, project entry points, and strict configuration validation.

Scope
Implementation of a unified `Makefile`, `uv`-based dependency locking, and a `Pydantic`-based configuration schema.

### Task [T1.1]: Create dependency manager and task runner

Type: Foundation

Description
Implement a standardized entry point for project tasks and lock dependencies.

Inputs
Existing scattered `requirements.txt` or loose environment files.

Outputs
A locked dependency file and a `Makefile` with standard targets.

```id="atomic-prompt-t1.1"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a reproducible dependency configuration and a universal task runner.

Context
Part of a larger system with the following requirement:
We need a deterministic way to install dependencies and run project commands (setup, run, test) without user friction.

Tasks
1. Define a dependency configuration file (e.g., `pyproject.toml` or `uv.lock`) that pins all required libraries (pydantic, docker, rich, streamlit, pytest, etc.).
2. Create a `Makefile` with the following targets: `install`, `setup`, `run`, and `test`.
3. Ensure the `install` target syncs the locked dependencies to the virtual environment.

Constraints
- Keep the Makefile POSIX compliant.
- Assume `uv` or standard `pip-compile` is the underlying package manager.

Safety Notes
- Fail clearly if the base python version is fundamentally incompatible.

Verification
- Provide a usage example demonstrating `make install` and `make run`.
```

### Task[T1.2]: Create strict configuration module

Type: Foundation

Description
Implement a centralized configuration parser that validates user inputs (data paths, dataset keys, random seeds).

Inputs
Raw JSON/YAML user configuration file.

Outputs
Validated configuration object usable across the system.

```id="atomic-prompt-t1.2"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a strict configuration parsing and validation module.

Context
Part of a larger system with the following requirement:
The system needs a validated configuration object defining dataset location, `batch_key`, `cell_type_key`, `random_seed`, and `selected_models`.

Tasks
1. Create a data structure (using Pydantic or standard Dataclasses) representing the system configuration.
2. Implement validation logic to ensure paths exist on the filesystem.
3. Allow `cell_type_key` to be nullable (for unsupervised-only runs) but enforce `batch_key` existence.
4. Set a default `random_seed` (e.g., 42) if none is provided.

Constraints
- Keep the module completely independent of machine learning logic.
- Ensure clear, human-readable error messages for missing required fields.

Safety Notes
- Validate external paths to prevent directory traversal or reading restricted files.

Verification
- Add a minimal test demonstrating validation success on valid input and clear exception on missing dataset paths.
```

---

## Sprint [S2]: Data Ingestion & Preprocessing

Goal
Implement the centralized data handling logic so datasets are processed exactly once before any models execute.

Scope
Dataset loader, column verifier, and single standardized output generator.

### Task [T2.1]: Implement dataset loader and validator

Type: Foundation

Description
Load the biological dataset and verify internal structural requirements based on the configuration.

Inputs
Validated configuration object from T1.2.

Outputs
In-memory representation of the dataset and extracted metadata (e.g., list of available omics).

```id="atomic-prompt-t2.1"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a dataset loader that verifies required internal structures.

Context
Part of a larger system with the following requirement:
Before running models, the system must confirm the dataset actually contains the biological metadata (batch, cell type) specified in the config.

Tasks
1. Create a module to load data files (h5ad/h5mu formats).
2. Implement logic to inspect the dataset headers/observations.
3. Verify that the `batch_key` and `cell_type_key` (if provided) actually exist in the dataset.
4. Extract and return a list of available omics (e.g.,["rna", "atac"]).

Constraints
- Do not modify the dataset in memory during validation.

Safety Notes
- Fail fast with a descriptive error if a configured key is missing from the dataset.

Verification
- Provide a usage example mocking a dataset load and successfully verifying keys.
```

---

# Epic [E2]: Core Feature - Model Registry & Dynamic Routing

Description
Decouple model rules from the codebase to allow dynamic matching of user datasets to compatible ML models.

Outcome
A declarative registry system that automatically filters and selects valid models based on the dataset's available omics.

---

## Sprint [S3]: Registry Engine

Goal
Build a declarative model registry and the dynamic routing logic.

Scope
Registry schema loader and the model matching engine.

### Task [T3.1]: Implement model registry schema

Type: Feature

Description
Create an externalized registry mapping models to their required Docker images and omics compatibility.

Inputs
None directly (standalone component).

Outputs
A loaded dictionary/object representing the available multiverse models.

```id="atomic-prompt-t3.1"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a static model registry loader.

Context
Part of a larger system with the following requirement:
The system needs to know which models exist, what Docker images they need, and what data modalities they support, without hardcoding this into Python logic.

Tasks
1. Define a YAML or JSON schema structure for a `model_registry`. It must include: model name, docker image tag, and a list of supported omics.
2. Create a loader function that reads this file and returns an indexed registry object.

Constraints
- Do not instantiate any Docker logic here. purely structural data loading.

Safety Notes
- Fail clearly if the registry file is missing or malformed.

Verification
- Provide a sample `registry.yaml` and a script demonstrating loading it into memory.
```

### Task [T3.2]: Implement dynamic routing logic

Type: Feature

Description
Match the extracted dataset omics to the available models in the registry to determine eligibility.

Inputs
Extracted omics list (from T2.1) and loaded registry (from T3.1).

Outputs
Filtered list of eligible models to be run.

```id="atomic-prompt-t3.2"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement dynamic matching logic for model eligibility.

Context
Part of a larger system with the following requirement:
Not all models support all data types. We must automatically filter out models from the user's selection that are incompatible with their dataset.

Tasks
1. Create a function `get_eligible_models(user_requested_models, available_omics, registry)`.
2. Implement intersection logic: check if the model's required omics are a subset or match the dataset's available omics.
3. Return the final list of models that are safe to execute.

Constraints
- Must handle edge cases (e.g., a model supports "any" omics vs strict lists).

Verification
- Add a minimal test passing a mock registry and asserting incompatible models are dropped from the output.
```

---

# Epic [E3]: Integration - Async Docker Orchestration

Description
Replace sequential, host-based execution with isolated, concurrent Docker container orchestration for maximum performance.

Outcome
The system can pull/build images concurrently and execute multiple ML models in parallel without blocking, while catching individual container failures gracefully.

---

## Sprint [S4]: Container Management

Goal
Build the asynchronous engine for Docker execution.

Scope
Parallel image builder, parallel model runner, and failure isolation.

### Task [T4.1]: Implement concurrent Docker image builder

Type: Integration

Description
Ensure all required Docker images for eligible models are built/pulled concurrently before execution starts.

Inputs
List of eligible models and their required image tags (from T3.2).

Outputs
Locally available, ready-to-run Docker images.

```id="atomic-prompt-t4.1"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement an asynchronous Docker image builder/puller.

Context
Part of a larger system with the following requirement:
To save time, the system must prepare all necessary Docker images concurrently before running any data pipelines.

Tasks
1. Create an async Python module utilizing a Docker client library.
2. Implement a function that takes a list of image tags and pulls/builds them concurrently using `asyncio.gather` or ThreadPoolExecutor.
3. Ensure thread-safe logging of build progress.

Constraints
- Do not start any model execution. Only ensure the images exist locally.

Safety Notes
- Catch Docker daemon connection errors immediately and halt.

Verification
- Provide a usage example that mocks the Docker client and demonstrates concurrent task execution.
```

### Task [T4.2]: Implement concurrent model runner with isolation

Type: Integration

Description
Execute eligible models in parallel via Docker, mounting the dataset as read-only, and tracking success/failure states.

Inputs
Eligible models, prepared dataset path, configuration (random seed).

Outputs
A status report dictionary mapping model names to success/failure states and output directories.

```id="atomic-prompt-t4.2"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement an asynchronous Docker execution engine with failure isolation.

Context
Part of a larger system with the following requirement:
Models must run concurrently. If one model fails (e.g., Out of Memory), the master process must catch it, record the failure, and allow the remaining parallel models to finish.

Tasks
1. Create an async function `run_models_concurrently(models, data_path, seed)`.
2. For each model, spin up a Docker container.
3. Mount the `data_path` as a read-only volume inside the container.
4. Inject the `seed` as an environment variable.
5. Wait for containers to exit and capture their exit codes.
6. Return a summary object indicating which models exited with code 0 (success) and which failed.

Constraints
- Ensure the master process does not crash if a container exits with a non-zero code.

Safety Notes
- Ensure volume mounts are strictly read-only for input data to prevent corruption.

Verification
- Add a minimal test mocking a container run where one container returns exit code 0 and another returns 1, verifying the master process survives and reports both correctly.
```

---

# Epic [E4]: Feature - Conditional Evaluation Engine

Description
Implement an intelligent downstream evaluation node that calculates metrics conditionally based on dataset properties and model success.

Outcome
An evaluation module that avoids crashing on missing data (e.g., trying to calculate batch effects when there's only one batch) and consolidates results into a single JSON.

---

## Sprint [S5]: Metrics Calculation

Goal
Build the conditional evaluator and result aggregator.

Scope
Metric toggle logic, condition checks, and result consolidation.

### Task[T5.1]: Implement conditional metrics logic

Type: Feature

Description
Determine which metrics are mathematically valid to run based on the initial data configuration.

Inputs
Validated configuration (T1.2) containing dataset metadata (number of batches, presence of cell types).

Outputs
A filtered list of metrics to compute.

```id="atomic-prompt-t5.1"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement conditional logic for evaluation metrics.

Context
Part of a larger system with the following requirement:
The system should only run supervised metrics (ARI, NMI) if a `cell_type_key` exists, and skip batch-correction metrics (Graph Connectivity) if only one batch is present in the dataset.

Tasks
1. Create a function `determine_valid_metrics(config, user_requested_metrics)`.
2. Implement rules: remove supervised metrics if `cell_type_key` is null.
3. Implement rules: remove batch metrics if the number of unique batches is 1.
4. Return the final actionable list of metrics.

Constraints
- Keep this logic distinct from the actual mathematical calculation of the metrics.

Verification
- Add a minimal test passing a config with `cell_type_key=None` and ensuring "ari" is removed from the requested metrics list.
```

### Task [T5.2]: Implement result aggregator

Type: Integration

Description
Iterate over the output directories of successful models, compute valid metrics, and write a unified JSON file.

Inputs
Model run status report (T4.2) and valid metrics list (T5.1).

Outputs
A final `results.json` in the save directory.

```id="atomic-prompt-t5.2"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a result aggregation module.

Context
Part of a larger system with the following requirement:
After models finish running, the system must collect outputs only from the successful ones, trigger metric calculations, and save everything into a single summary file.

Tasks
1. Create a function that iterates over the model status report.
2. Ignore models marked as failed.
3. For successful models, simulate/trigger the computation of the provided valid metrics list.
4. Aggregate the final scores into a single dictionary and serialize it to `results.json` in a specified output directory.

Constraints
- Must handle file I/O safely and overwrite existing results cleanly.

Verification
- Provide a usage example with dummy model scores writing to a temporary JSON file.
```

---

# Epic [E5]: User Experience & Hardening

Description
Provide intuitive interfaces (CLI dashboards, GUI wizards) and robust documentation so "rookie" users can operate the system flawlessly.

Outcome
A polished CLI with live progress bars, a web-based setup wizard, and compiled static documentation.

---

## Sprint [S6]: Interfaces

Goal
Build the terminal dashboard and the GUI setup wizard.

Scope
Rich CLI integration and Streamlit GUI.

### Task [T6.1]: Implement live CLI dashboard

Type: Feature

Description
Wrap the orchestrator in a visual CLI layer that displays real-time progress for parallel Docker tasks.

Inputs
Async state from T4.1 and T4.2.

Outputs
Visual terminal interface showing active, completed, and failed tasks.

```id="atomic-prompt-t6.1"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a live terminal dashboard for parallel tasks.

Context
Part of a larger system with the following requirement:
Users need visual feedback while models run in parallel, showing which are building, running, successful, or failed.

Tasks
1. Use a terminal formatting library (e.g., `Rich`).
2. Implement a `Live` table or `Progress` bar layout.
3. Create callback functions or state updaters that the Docker runner can call to update the status of a specific model from "Pending" to "Running" to "Done/Failed".

Constraints
- The UI layer must not block the async execution loop.

Verification
- Provide a standalone script simulating parallel sleep tasks updating a live terminal table.
```

### Task [T6.2]: Implement interactive setup GUI

Type: Feature

Description
Create a lightweight web application that allows users to configure their run without editing JSON/YAML files.

Inputs
None directly (acts as a frontend generator).

Outputs
A generated configuration file ready for the core pipeline.

```id="atomic-prompt-t6.2"
Implementation Prompt

Act as a senior software engineer.

Goal
Implement a lightweight setup GUI.

Context
Part of a larger system with the following requirement:
Bioinformaticians need a simple web interface to point to their dataset, select models, and generate the system configuration file.

Tasks
1. Create a basic web script (e.g., using `Streamlit`).
2. Add form inputs for: Dataset Path (text), Cell Type Key (text/optional), Batch Key (text), Output Directory (text).
3. Add a multi-select box for available models.
4. On submit, serialize the inputs into a valid YAML/JSON configuration file matching the schema from T1.2.

Constraints
- Keep it entirely contained in one file.
- Do not trigger the pipeline execution from here; only generate the config file.

Verification
- Provide the Python script structure demonstrating the layout and file-saving logic.
```