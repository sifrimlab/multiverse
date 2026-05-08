# Model Container Contract

This document defines the language-agnostic runtime contract between the orchestrator and model containers.

## Mount Points

- Input mount (read-only): `/input/data.h5mu`
- Output mount (read-write): `/output/`

The orchestrator must mount only these paths for model execution.

## Job Spec Contract

The orchestrator writes a JSON file at:

- `/output/job_spec.json`

Expected top-level keys:

- `seed` (integer)
- `dataset_id` (integer or null)
- `dataset_name` (string)
- `model_name` (string)
- `hyperparameters` (object)
- `run_settings` (object)

Containers should treat this file as the canonical runtime configuration source.

## Expected Output Artifacts

Containers are expected to write the following files into `/output/`:

- `embeddings.h5` (required): latent embeddings in HDF5 format
- `metrics.json` (required): model metrics and diagnostics
- `container.log` (written by orchestrator): captured stdout/stderr for the run

Optional artifacts (for visualization/debugging) may also be written, but must not replace required files.

## Container Behavior Rules

- Container code must not depend on host filesystem layouts.
- Container must read input only from `/input/data.h5mu`.
- Container must write outputs only under `/output/`.
- Container must not attempt Docker registry push operations.
