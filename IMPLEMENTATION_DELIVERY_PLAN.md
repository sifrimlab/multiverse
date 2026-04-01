# Epic [E1]: State Management & The Registry Architecture

Description
Transition the system from reading static JSON configuration files (`config_alldatasets.json`) to a dynamic SQLite-backed Registry. This enables idempotency (skipping completed runs) and decoupled execution.

Outcome
A local SQLite database tracking Datasets, Models, and Runs, along with a structured physical file Store.

---

## Sprint [S1]: Database & Ingestion Layer

Goal
Create the SQLite schema and update the dataset ingestion process to register datasets into the database instead of a JSON file.

Scope
SQLite database initialization, Store directory creation, and an ingestion script.

### Task[T1.1]: Initialize local state database and store directories

Type: Foundation

Description
Create the `mvexp_state.db` SQLite database with tables for Datasets, Models, and Runs, and initialize the physical `store/` directories.

Inputs
None.

Outputs
A new module `multiverse/registry_db.py` and directory structure.

Implementation Prompt

Act as a senior software engineer.

Goal
Implement a lightweight SQLite state manager and directory initializer.

Context
Part of a larger system with the following requirement:
We are migrating from static JSON configs to a local SQLite registry to track execution state. We need a module to initialize the DB schema and create the required physical folders.

Tasks
1. Create `multiverse/registry_db.py`.
2. Implement an initialization function that creates `store/datasets/`, `store/models/`, and `store/artifacts/`.
3. Implement SQLite table creation using the built-in `sqlite3` library:
   - `datasets` (id, name, path, omics_available, status)
   - `models` (name, docker_image, supported_omics)
   - `runs` (run_id, dataset_id, model_name, status, output_path)
4. Ensure the initialization is idempotent (using `CREATE TABLE IF NOT EXISTS`).

Constraints
- Do not use ORMs like SQLAlchemy; keep it strictly minimal with standard library `sqlite3`.
- Do not hardcode absolute paths; use relative paths from the project root.

Verification
- Provide a Python snippet demonstrating the initialization of the DB and inserting a dummy dataset row.


### Task[T1.2]: Create dataset registration CLI command

Type: Feature

Description
Update `multiverse/ingestion.py` and `multiverse/dataloader.py` so that when a user registers a dataset, it validates the structure and inserts the metadata into the SQLite registry.

Inputs
`multiverse/ingestion.py`, `multiverse/registry_db.py`.

Outputs
A command-line entry point to register a dataset.

Implementation Prompt

Act as a senior software engineer.

Goal
Implement a CLI command to register datasets into the SQLite registry.

Context
Part of a larger system with the following requirement:
Instead of defining datasets in `config.json`, users will run a command to register a dataset. The system must validate the dataset and save its metadata to the DB.

Tasks
1. Create a function `register_dataset(file_path: str, name: str, batch_key: str)` in `multiverse/ingestion.py`.
2. Use existing `validate_dataset_structure` to extract available omics (e.g., `["rna", "atac"]`).
3. Copy the dataset into `store/datasets/raw/`.
4. Insert a record into the `datasets` SQLite table with status="READY" and the extracted omics.

Constraints
- Fail gracefully if the dataset is corrupted or missing the required `batch_key`.

Verification
- Add a minimal unit test mocking the file copy and verifying the correct SQLite `INSERT` statement is generated.

---

# Epic[E2]: Orchestrator Hardening & Idempotency

Description
Update the Docker runner to use the new SQLite registry for planning runs, enforcing resource guardrails, and tracking execution state.

Outcome
The `docker_runner.py` is fully idempotent, respects concurrency limits, and isolates file permissions.

---

## Sprint [S2]: The Execution Engine

Goal
Refactor the async Docker runner to read from the DB, protect host resources, and handle failures gracefully.

Scope
Updates to `multiverse/runner/docker_runner.py` and `multiverse/runner/cli.py`.

### Task [T2.1]: Implement run planning and idempotency

Type: Feature

Description
Modify the orchestrator to check the SQLite database for existing successful runs before spinning up Docker containers.

Inputs
`multiverse/registry_db.py`, `model_registry.json`.

Outputs
A "Planner" function that generates a list of required jobs.

Implementation Prompt

Act as a senior software engineer.

Goal
Implement an execution planner that calculates the delta of required ML jobs.

Context
Part of a larger system with the following requirement:
If a user runs the benchmark command, the system should check the DB for all "READY" datasets and compatible models. It must skip any dataset+model combination that already has a "SUCCESS" status in the `runs` table.

Tasks
1. In `multiverse/runner/cli.py`, create a function `generate_execution_plan(db_connection)`.
2. Query the DB to cross-reference eligible models (based on dataset omics).
3. Filter out any combinations where `status == 'SUCCESS'` in the `runs` table.
4. Return a list of dictionary objects representing the pending jobs.

Constraints
- Must handle cases where previous runs failed (status == 'FAILED'); these SHOULD be retried.

Verification
- Provide a mock DB state and show the function returning only the pending/failed jobs.


### Task [T2.2]: Apply Docker resource guardrails and permissions

Type: Hardening

Description
Update `run_models_concurrently` and `run_model_container` in `docker_runner.py` to enforce file ownership mappings and memory limits.

Inputs
`multiverse/runner/docker_runner.py`.

Outputs
Hardened Docker run configurations.

Implementation Prompt

Act as a senior software engineer.

Goal
Harden the Docker execution layer with resource limits and UID mapping.

Context
Part of a larger system with the following requirement:
Containers currently run as root, creating output files the host user cannot delete. Furthermore, uncontrolled containers might cause an OS out-of-memory error.

Tasks
1. Modify `run_model_container` in `multiverse/runner/docker_runner.py`.
2. Inject the host machine's User ID (UID) and Group ID (GID) into the Docker run parameters using the `user` argument in `docker-py` (e.g., `user=f"{os.getuid()}:{os.getgid()}"`).
3. Add a memory limit parameter to the container run config (e.g., `mem_limit='16g'`).
4. Mount the input dataset volume as strictly read-only (`mode='ro'`).

Constraints
- Ensure compatibility with both Linux and macOS UID fetching logic.

Verification
- Provide the updated Python function definition showing the new `docker.containers.run` arguments.

---

# Epic [E3]: Decoupled Evaluation & Output Contracts

Description
Enforce strict I/O contracts on `ModelFactory` subclasses and decouple the `scIB-metrics` evaluator into an independent background process.

Outcome
Models write outputs atomically, and the evaluator watches the artifact store to continuously update results.

---

## Sprint [S3]: Atomic Outputs & Evaluator Refactoring

Goal
Ensure model outputs are safe from corruption and build the standalone evaluator script.

Scope
Updates to `multiverse/models/base.py` and `multiverse/evaluate.py`.

### Task[T3.1]: Implement atomic writes for model latent saves

Type: Hardening

Description
Update `ModelFactory.save_latent()` to write to a temporary file before renaming it, preventing corrupted data if a model crashes.

Inputs
`multiverse/models/base.py`.

Outputs
Resilient `save_latent` method.

Implementation Prompt

Act as a senior software engineer.

Goal
Implement atomic file writes for model embeddings.

Context
Part of a larger system with the following requirement:
If a model container crashes while saving `embeddings.h5`, a corrupted file is left behind, tricking the system into thinking the run succeeded. We must ensure writes are atomic.

Tasks
1. Modify `save_latent()` in `multiverse/models/base.py`.
2. Change the output file path to use a `.tmp` suffix during the `h5py.File` write operation.
3. Upon successful closure of the file, use `os.rename` or `shutil.move` to remove the `.tmp` suffix.

Constraints
- Must work seamlessly across all subclasses (PCA, MOFA, Mowgli, etc.).

Verification
- Add a minimal unit test simulating a successful save and verifying the `.tmp` file is renamed correctly.


### Task [T3.2]: Refactor Evaluator for decoupled execution

Type: Integration

Description
Modify `multiverse/evaluate.py` to accept specific run IDs from the SQLite DB rather than reading a master config file, allowing it to run independently on newly finished outputs.

Inputs
`multiverse/evaluate.py`, `multiverse/registry_db.py`.

Outputs
A standalone `evaluate_run()` function.

Implementation Prompt

Act as a senior software engineer.

Goal
Refactor the evaluation module to process single completed runs dynamically.

Context
Part of a larger system with the following requirement:
The `evaluate.py` script currently evaluates *all* models based on a central config. It needs to be refactored so a watcher daemon can pass it a specific `run_id` and output directory as soon as a single model finishes.

Tasks
1. Modify `multiverse/evaluate.py`.
2. Create a function `evaluate_single_run(output_dir: str, dataset_path: str, batch_key: str, label_key: str)`.
3. Load the specific model's `embeddings.h5` from `output_dir`.
4. Run `scib_metrics` and save the resulting JSON strictly inside that `output_dir`.

Constraints
- Remove dependencies on the monolithic `config_alldatasets.json`.
- Do not crash if `label_key` is missing; calculate unsupervised metrics only.

Verification
- Provide the updated function signature and logic flow for `evaluate_single_run`.
