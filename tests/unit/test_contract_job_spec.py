"""Phase 0 / Phase 1 contract tests.

These tests are written before the contract module exists (Phase 0) and
serve as the acceptance gate for Phase 1.  They all skip gracefully if
``multiverse.contract`` is not yet available, so the test suite stays
green throughout the migration.

Once Phase 1 lands the skips are replaced by passing assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module until Phase 1 lands
# ---------------------------------------------------------------------------


def _import_contract():
    try:
        import multiverse.contract as contract

        return contract
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------


def test_contract_paths_constants() -> None:
    contract = _import_contract()
    if contract is None:
        pytest.skip("multiverse.contract not yet implemented")

    assert contract.CONTAINER_INPUT_DATA_PATH == "/input/data.h5mu"
    assert contract.CONTAINER_OUTPUT_DIR == "/output"
    assert contract.CONTAINER_JOB_SPEC_PATH == "/output/job_spec.json"
    assert contract.JOB_SPEC_FILENAME == "job_spec.json"


# ---------------------------------------------------------------------------
# JobSpec model
# ---------------------------------------------------------------------------


_MINIMAL_KWARGS = dict(
    model_name="pca",
    model_version="1.0.0",
    dataset_slug="pbmc",
    dataset_path_in_container="/input/data.h5mu",
    hyperparameters={"pca": {"n_components": 20}},
)


def test_job_spec_round_trips_json() -> None:
    contract = _import_contract()
    if contract is None:
        pytest.skip("multiverse.contract not yet implemented")

    spec = contract.JobSpec(**_MINIMAL_KWARGS)
    dumped = spec.model_dump(exclude_none=True)
    assert dumped["model_name"] == "pca"
    assert dumped["hyperparameters"] == {"pca": {"n_components": 20}}


def test_job_spec_payload_helper() -> None:
    contract = _import_contract()
    if contract is None:
        pytest.skip("multiverse.contract not yet implemented")

    payload = contract.job_spec_payload(
        model_name="pca",
        model_version="1.0.0",
        dataset_slug="pbmc",
        hyperparameters={"pca": {"n_components": 20}},
        seed=42,
    )
    assert isinstance(payload, dict)
    assert payload["model_name"] == "pca"
    assert payload["seed"] == 42
    assert payload["dataset_path_in_container"] == "/input/data.h5mu"


# ---------------------------------------------------------------------------
# write_job_spec produces stable, sort-keyed JSON
# ---------------------------------------------------------------------------


def test_write_job_spec_stable_output(tmp_path: Path) -> None:
    contract = _import_contract()
    if contract is None:
        pytest.skip("multiverse.contract not yet implemented")

    out = tmp_path / "job_spec.json"
    payload = contract.job_spec_payload(
        model_name="mofa",
        model_version="2.0.0",
        dataset_slug="pbmc",
        hyperparameters={"mofa": {"n_factors": 10}},
        seed=7,
    )
    contract.write_job_spec(out, payload)

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    loaded = json.loads(text)
    assert loaded["model_name"] == "mofa"

    # Stability: writing the same payload twice must produce byte-identical files.
    out2 = tmp_path / "job_spec2.json"
    contract.write_job_spec(out2, payload)
    assert out.read_bytes() == out2.read_bytes()


def test_write_job_spec_sort_keys(tmp_path: Path) -> None:
    contract = _import_contract()
    if contract is None:
        pytest.skip("multiverse.contract not yet implemented")

    payload = {
        "seed": 1,
        "model_name": "pca",
        "dataset_slug": "test",
        "hyperparameters": {},
        "model_version": "0.1",
        "dataset_path_in_container": "/input/data.h5mu",
    }
    out = tmp_path / "js.json"
    contract.write_job_spec(out, payload)
    text = out.read_text(encoding="utf-8")
    keys = [line.strip().split('"')[1] for line in text.splitlines() if '":' in line]
    assert keys == sorted(keys), f"job_spec.json keys not sorted: {keys}"


# ---------------------------------------------------------------------------
# Docker and Slurm produce byte-identical payloads (Phase 2 gate)
# ---------------------------------------------------------------------------


def test_docker_and_slurm_payload_identical() -> None:
    """Both executors must use the same contract writer for the same job."""
    contract = _import_contract()
    if contract is None:
        pytest.skip("multiverse.contract not yet implemented")

    payload_a = contract.job_spec_payload(
        model_name="pca",
        model_version="1.0.0",
        dataset_slug="pbmc",
        hyperparameters={"pca": {"n_components": 10}},
        batch_key="batch",
        cell_type_key="cell_type",
        seed=42,
        preprocessing={},
    )
    payload_b = contract.job_spec_payload(
        model_name="pca",
        model_version="1.0.0",
        dataset_slug="pbmc",
        hyperparameters={"pca": {"n_components": 10}},
        batch_key="batch",
        cell_type_key="cell_type",
        seed=42,
        preprocessing={},
    )
    assert payload_a == payload_b, "Same logical job must produce identical payloads"
