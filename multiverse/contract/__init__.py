"""multiverse.contract — dependency-clean file-contract primitives.

Imports from this package are safe in any environment: no Docker, no ML
frameworks, no scientific I/O libraries.  Only pydantic and the Python
standard library are required.

Public surface
--------------
- :data:`CONTAINER_INPUT_DATA_PATH`
- :data:`CONTAINER_OUTPUT_DIR`
- :data:`CONTAINER_JOB_SPEC_PATH`
- :data:`JOB_SPEC_FILENAME`
- :class:`JobSpec`
- :func:`job_spec_payload`
- :func:`write_job_spec`
"""

from .job_spec import JobSpec, job_spec_payload, write_job_spec
from .paths import (
    CONTAINER_INPUT_DATA_PATH,
    CONTAINER_JOB_SPEC_PATH,
    CONTAINER_OUTPUT_DIR,
    ENV_INPUT_DATA_PATH,
    ENV_JOB_SPEC_PATH,
    ENV_OUTPUT_DIR,
    JOB_SPEC_FILENAME,
)

__all__ = [
    "CONTAINER_INPUT_DATA_PATH",
    "CONTAINER_JOB_SPEC_PATH",
    "CONTAINER_OUTPUT_DIR",
    "ENV_INPUT_DATA_PATH",
    "ENV_JOB_SPEC_PATH",
    "ENV_OUTPUT_DIR",
    "JOB_SPEC_FILENAME",
    "JobSpec",
    "job_spec_payload",
    "write_job_spec",
]
