"""sbatch template golden-shape tests (STRATEGY M4).

The template is deterministic given the spec; a drift in directive
ordering or shell quoting must surface as a test diff, not a silent
behavioural change on a cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.slurm import SlurmJobSpec, render_sbatch_script


pytestmark = pytest.mark.control_plane


def _spec(**overrides) -> SlurmJobSpec:
    base = dict(
        job_name="mvd-test",
        image_sif=Path("/sif/model.sif"),
        workspace=Path("/work/abc"),
        dataset_path=Path("/data/x.h5mu"),
        command=["python", "-m", "model.run"],
        env={"MVR_OUTPUT_DIR": "/output", "FOO": "bar baz"},
        cpus_per_task=4,
    )
    base.update(overrides)
    return SlurmJobSpec(**base)


def test_minimal_script_has_required_directives() -> None:
    script = render_sbatch_script(_spec())
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=mvd-test" in lines
    assert "#SBATCH --cpus-per-task=4" in lines


def test_partition_account_qos_emitted_when_set() -> None:
    script = render_sbatch_script(
        _spec(partition="gpu", account="lab1", qos="premium")
    )
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=lab1" in script
    assert "#SBATCH --qos=premium" in script


def test_time_and_mem_directives() -> None:
    script = render_sbatch_script(_spec(time_minutes=120, mem_gb=32))
    assert "#SBATCH --time=120" in script
    assert "#SBATCH --mem=32G" in script


def test_gpu_directive_emits_gres() -> None:
    script = render_sbatch_script(_spec(gpus=2))
    assert "#SBATCH --gres=gpu:2" in script


def test_extra_directives_passthrough() -> None:
    script = render_sbatch_script(
        _spec(extra_directives=["--exclusive", "#SBATCH --constraint=mlnx"])
    )
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --constraint=mlnx" in script


def test_env_is_sorted_and_quoted() -> None:
    script = render_sbatch_script(_spec())
    # Both env vars appear in alphabetical order.
    foo_idx = script.find("export FOO=")
    out_idx = script.find("export MVR_OUTPUT_DIR=")
    assert foo_idx >= 0 and out_idx >= 0
    assert foo_idx < out_idx
    # Spaces are quoted.
    assert "export FOO='bar baz'" in script


def test_apptainer_command_binds_dataset_ro_and_workspace_rw() -> None:
    script = render_sbatch_script(_spec())
    assert "--bind /data/x.h5mu:/input/data.h5mu:ro" in script
    assert "--bind /work/abc:/output:rw" in script
    assert "/sif/model.sif" in script
    # User command is preserved.
    assert "python -m model.run" in script


def test_paths_with_spaces_are_quoted() -> None:
    script = render_sbatch_script(
        _spec(
            image_sif=Path("/sif dir/model.sif"),
            workspace=Path("/work dir/abc"),
            dataset_path=Path("/data dir/x.h5mu"),
        )
    )
    # No raw unquoted space appears in the bind paths.
    assert "'/sif dir/model.sif'" in script
    assert "'/work dir/abc'" in script
    assert "'/data dir/x.h5mu'" in script
