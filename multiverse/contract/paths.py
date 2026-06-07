"""Canonical container paths and environment variable defaults.

This module is dependency-clean: stdlib only, no scientific I/O, no Docker,
no ML frameworks.  It is safe to import in any environment.
"""

from __future__ import annotations

# Container contract: model code only ever sees these fixed in-container
# paths, never host paths. ``/input/data.h5mu`` is mounted read-only; the
# job spec and all results are written under the writable ``/output``. This
# decoupling is what lets the same image run unchanged under Docker or Slurm.
CONTAINER_INPUT_DATA_PATH: str = "/input/data.h5mu"
CONTAINER_OUTPUT_DIR: str = "/output"
CONTAINER_JOB_SPEC_PATH: str = "/output/job_spec.json"
JOB_SPEC_FILENAME: str = "job_spec.json"

# Environment variable names used by both the host executor and the container
# worker SDK.
ENV_INPUT_DATA_PATH: str = "MVR_INPUT_DATA_PATH"
ENV_OUTPUT_DIR: str = "MVR_OUTPUT_DIR"
ENV_JOB_SPEC_PATH: str = "MVR_JOB_SPEC_PATH"
