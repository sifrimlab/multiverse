import json
import os
from typing import Any, Dict

import mudata as md

from ..logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

INPUT_DATA_PATH = "/input/data.h5mu"
OUTPUT_DIR = "/output"
JOB_SPEC_PATH = "/output/job_spec.json"


def load_job_spec(job_spec_path: str = JOB_SPEC_PATH) -> Dict[str, Any]:
    """Load the per-run job specification written by the orchestrator."""
    with open(job_spec_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def setup_container_logging(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)


def load_input_mudata(input_path: str = INPUT_DATA_PATH) -> md.MuData:
    """Load canonical container input dataset from a fixed mount path."""
    return md.read_h5mu(input_path)


def build_model_config(model_name: str, job_spec: Dict[str, Any], output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """Build a minimal in-memory config consumed by ModelFactory subclasses."""
    model_params = job_spec.get("hyperparameters", {})
    if model_name in model_params:
        scoped_model_params = {model_name: model_params.get(model_name, {})}
    else:
        scoped_model_params = {model_name: model_params}

    return {
        "output_dir": output_dir,
        "seed": job_spec.get("seed"),
        "model": scoped_model_params,
        "metrics": job_spec.get("metrics", {}),
    }
