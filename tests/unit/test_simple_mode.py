"""Milestone-3 exit-gate tests for the simple-mode runner.

These tests use the synthetic backend and do not require Docker. They cover:

    1. Happy path: a contract-valid producer yields a verified bundle.
    2. Validator refusal: a bad producer yields a run_attempt_manifest with
       state EVALUATION_FAILED, preserved workspace, and a recovery hint.
    3. Strict mode: an ``unverified_local`` image identity is refused before
       validation runs.
    4. Logical run ID is stable when the same recipe is executed twice.
    5. CLI argparse builds; ``--strict`` and ``--validators`` round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from multiverse.artifact import (
    ARTIFACT_MANIFEST_FILENAME,
    ImageIdentity,
    read_manifest,
)
from multiverse.simple import (
    JobOutcome,
    SimpleModeRunner,
    SyntheticBackend,
    parse_simple_manifest,
)
from multiverse.simple.cli import build_parser
from multiverse.simple.runner import JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest_file(tmp_path: Path, n_obs: int = 4,
                         digest: str | None = "sha256:" + "a" * 64) -> Path:
    digest_line = f'      image_digest: "{digest}"\n' if digest else ""
    text = (
        'schema_version: "1"\n'
        'globals:\n'
        '  mv_contract_version: "1"\n'
        'jobs:\n'
        '  - name: "demo_pca"\n'
        '    model:\n'
        '      slug: "pca"\n'
        '      version: "1.0.0"\n'
        '      image: "multiverse-pca:1.0.0"\n'
        f'{digest_line}'
        '      contract_version: "1"\n'
        '    dataset:\n'
        '      slug: "demo"\n'
        '      path: "/tmp/nonexistent/demo.h5mu"\n'
        f'      n_obs: {n_obs}\n'
        '      n_vars: 32\n'
        '    params:\n'
        '      n_components: 4\n'
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _good_producer(n_obs: int):
    """Returns a producer that writes a contract-valid embeddings.h5."""

    def _producer(workspace: Path, job: Any) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            arr = np.random.default_rng(0).standard_normal((n_obs, 4)).astype(
                np.float32
            )
            f.create_dataset("latent", data=arr)

    return _producer


def _bad_n_obs_producer(actual_n_obs: int):
    def _producer(workspace: Path, job: Any) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            arr = np.zeros((actual_n_obs, 4), dtype=np.float32)
            f.create_dataset("latent", data=arr)

    return _producer


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_produces_verified_bundle(tmp_path: Path) -> None:
    manifest_path = _write_manifest_file(tmp_path, n_obs=4)
    manifest = parse_simple_manifest(manifest_path)
    out = tmp_path / "out"
    runner = SimpleModeRunner(
        backend=SyntheticBackend(producer=_good_producer(4)),
        output_root=out,
    )

    result = runner.run(manifest)
    assert result.all_succeeded, result.outcomes

    outcome: JobOutcome = result.outcomes[0]
    assert outcome.status is JobStatus.ARTIFACT_SUCCESS
    bundle = outcome.bundle_path
    assert bundle is not None and bundle.is_dir()

    # Verified read against the sidecar — proves the bundle is contract-valid.
    loaded = read_manifest(bundle)
    assert loaded.logical_run_id == outcome.logical_run_id
    assert {a.name for a in loaded.artifacts} == {"embeddings.h5"}
    assert (bundle / "outputs" / "embeddings.h5").is_file()
    assert (bundle / "manifest.txt").is_file()
    assert (bundle / "validation_report.json").is_file()
    env = json.loads((bundle / "environment.json").read_text())
    assert env["validators"] == "basic"
    assert env["strict"] is False
    assert env["backend"] == "synthetic"


def test_state_transitions_are_recorded_in_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest_file(tmp_path, n_obs=4)
    manifest = parse_simple_manifest(manifest_path)
    runner = SimpleModeRunner(
        backend=SyntheticBackend(producer=_good_producer(4)),
        output_root=tmp_path / "out",
    )
    result = runner.run(manifest)
    loaded = read_manifest(result.outcomes[0].bundle_path)
    transitions = [(t.from_state, t.to_state) for t in loaded.state_transitions]
    assert ("PENDING", "RUNNING") in transitions
    assert ("RUNNING", "TRAINING_SUCCEEDED") in transitions
    assert ("TRAINING_SUCCEEDED", "EVALUATING") in transitions
    assert ("EVALUATING", "ARTIFACT_SUCCESS") in transitions


# ---------------------------------------------------------------------------
# 2. Validator refusal → EVALUATION_FAILED
# ---------------------------------------------------------------------------


def test_wrong_n_obs_produces_failed_attempt_with_preserved_workspace(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_file(tmp_path, n_obs=4)
    manifest = parse_simple_manifest(manifest_path)
    out = tmp_path / "out"
    runner = SimpleModeRunner(
        backend=SyntheticBackend(producer=_bad_n_obs_producer(8)),
        output_root=out,
    )
    result = runner.run(manifest)

    assert not result.all_succeeded
    outcome = result.outcomes[0]
    assert outcome.status is JobStatus.EVALUATION_FAILED
    assert outcome.failure_dir and outcome.failure_dir.is_dir()
    # Preserved workspace under failure_dir/workspace/ (S5).
    preserved = outcome.failure_dir / "workspace" / "embeddings.h5"
    assert preserved.is_file()
    attempt = json.loads(
        (outcome.failure_dir / "run_attempt_manifest.json").read_text()
    )
    assert attempt["final_state"] == "EVALUATION_FAILED"
    assert attempt["validation_report"]["passed"] is False
    assert any(
        i["code"] == "EMBEDDING_WRONG_N_OBS"
        for i in attempt["validation_report"]["issues"]
    )
    # No bundle was promoted.
    assert not (out / "demo_pca" / ARTIFACT_MANIFEST_FILENAME).exists()


def test_backend_exception_yields_FAILED(tmp_path: Path) -> None:
    manifest_path = _write_manifest_file(tmp_path, n_obs=4)
    manifest = parse_simple_manifest(manifest_path)

    def _exploding(workspace: Path, job: Any) -> None:
        raise RuntimeError("simulated container crash")

    runner = SimpleModeRunner(
        backend=SyntheticBackend(producer=_exploding),
        output_root=tmp_path / "out",
    )
    result = runner.run(manifest)
    outcome = result.outcomes[0]
    assert outcome.status is JobStatus.FAILED
    attempt = json.loads(
        (outcome.failure_dir / "run_attempt_manifest.json").read_text()
    )
    assert attempt["final_state"] == "FAILED"
    assert "simulated container crash" in attempt["failure_reason"]


# ---------------------------------------------------------------------------
# 3. Strict-mode image-identity gate (R10 acceptance)
# ---------------------------------------------------------------------------


def test_strict_mode_refuses_unverified_local(tmp_path: Path) -> None:
    # Manifest has no digest → SyntheticBackend reports unverified_local.
    manifest_path = _write_manifest_file(tmp_path, n_obs=4, digest=None)
    manifest = parse_simple_manifest(manifest_path)
    runner = SimpleModeRunner(
        backend=SyntheticBackend(producer=_good_producer(4)),
        output_root=tmp_path / "out",
        strict=True,
    )
    result = runner.run(manifest)
    assert not result.all_succeeded
    outcome = result.outcomes[0]
    assert outcome.status is JobStatus.FAILED
    assert "strict mode refused image identity variant" in outcome.failure_reason
    attempt = json.loads(
        (outcome.failure_dir / "run_attempt_manifest.json").read_text()
    )
    assert attempt["image_identity"]["kind"] == "unverified_local"


def test_strict_mode_accepts_registry_digest(tmp_path: Path) -> None:
    manifest_path = _write_manifest_file(tmp_path, n_obs=4,
                                         digest="sha256:" + "b" * 64)
    manifest = parse_simple_manifest(manifest_path)
    runner = SimpleModeRunner(
        backend=SyntheticBackend(producer=_good_producer(4)),
        output_root=tmp_path / "out",
        strict=True,
    )
    result = runner.run(manifest)
    assert result.all_succeeded
    loaded = read_manifest(result.outcomes[0].bundle_path)
    assert loaded.image_identity.kind == ImageIdentity.registry_digest(
        "sha256:" + "b" * 64
    ).kind


# ---------------------------------------------------------------------------
# 4. Logical run ID stability across runs (S16 acceptance)
# ---------------------------------------------------------------------------


def test_logical_run_id_is_stable_across_runs(tmp_path: Path) -> None:
    manifest_path = _write_manifest_file(tmp_path, n_obs=4,
                                         digest="sha256:" + "d" * 64)
    manifest = parse_simple_manifest(manifest_path)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    runner_a = SimpleModeRunner(
        backend=SyntheticBackend(producer=_good_producer(4)),
        output_root=out_a,
    )
    runner_b = SimpleModeRunner(
        backend=SyntheticBackend(producer=_good_producer(4)),
        output_root=out_b,
    )
    a = runner_a.run(manifest)
    b = runner_b.run(manifest)
    assert a.outcomes[0].logical_run_id == b.outcomes[0].logical_run_id
    # But physical attempt IDs differ.
    assert a.outcomes[0].physical_attempt_id != b.outcomes[0].physical_attempt_id


# ---------------------------------------------------------------------------
# 5. CLI parser shape (R7)
# ---------------------------------------------------------------------------


def test_cli_parser_has_R7_flags(tmp_path: Path) -> None:
    parser = build_parser()
    ns = parser.parse_args(
        [
            str(tmp_path / "m.yaml"),
            "--out",
            str(tmp_path / "out"),
            "--strict",
            "--validators",
            "strict",
            "--no-image-pull",
            "--seed",
            "7",
            "--json",
        ]
    )
    assert ns.strict is True
    assert ns.validators == "strict"
    assert ns.no_image_pull is True
    assert ns.seed == 7
    assert ns.json is True


def test_cli_rejects_missing_required_args() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------------------------------------------------------------------------
# 6. Manifest parser
# ---------------------------------------------------------------------------


def test_manifest_parser_rejects_missing_n_obs(tmp_path: Path) -> None:
    bad = tmp_path / "m.yaml"
    bad.write_text(
        """
schema_version: "1"
jobs:
  - name: x
    model: {slug: pca, image: foo}
    dataset: {slug: d, path: /tmp/x}
""",
        encoding="utf-8",
    )
    from multiverse.simple.manifest import SimpleManifestError

    with pytest.raises(SimpleManifestError) as exc:
        parse_simple_manifest(bad)
    assert "n_obs" in str(exc.value)


def test_manifest_parser_accepts_minimal_valid_manifest(tmp_path: Path) -> None:
    good = tmp_path / "m.yaml"
    good.write_text(
        """
schema_version: "1"
jobs:
  - name: x
    model: {slug: pca, image: foo}
    dataset: {slug: d, path: /tmp/x, n_obs: 10}
""",
        encoding="utf-8",
    )
    parsed = parse_simple_manifest(good)
    assert len(parsed.jobs) == 1
    assert parsed.jobs[0].dataset_n_obs == 10
    assert parsed.jobs[0].params == {}
    assert parsed.jobs[0].validators == "basic"
