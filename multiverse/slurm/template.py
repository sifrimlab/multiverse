"""sbatch script generation (STRATEGY M4).

A run dispatched via the Slurm executor is one ``sbatch`` invocation per
attempt. The script body wraps a single ``apptainer exec`` call against
a pre-built SIF (preferred) or a runtime-pulled SIF.

We *do not* invent a job-script DSL. The template takes a small set of
named knobs (partition, account, qos, time/mem/cpus, plus a free-form
``extra_directives`` escape hatch for the site-specific incantations
every cluster has). Users who need more bend the template, not us.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..contract import JOB_SPEC_FILENAME


@dataclass(frozen=True)
class SlurmJobSpec:
    """Inputs the executor hands to a :class:`SlurmEngine`.

    ``image_sif`` is the path to the SIF that the apptainer step will
    execute. ``oci_digest`` (and the SIF's own digest, recorded later by
    the manifest's runtime_image_identity) is what enforces the M2
    dual-digest invariant.
    """

    job_name: str
    image_sif: Path
    workspace: Path
    dataset_path: Path
    command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    time_minutes: Optional[int] = None
    mem_gb: Optional[int] = None
    cpus_per_task: int = 1
    gpus: Optional[int] = None
    extra_directives: List[str] = field(default_factory=list)
    """Raw ``#SBATCH`` lines appended after the structured directives.
    Each entry must include the leading ``--`` (e.g. ``--gres=gpu:1``)
    and *not* the ``#SBATCH `` prefix."""

    output_log: Optional[Path] = None
    error_log: Optional[Path] = None
    apptainer_bin: str = "apptainer"
    use_tmpdir: bool = False
    """Stage large inputs (and bind ``/output``) on node-local
    ``$SLURM_TMPDIR`` instead of the shared filesystem, copying results
    back into the workspace as the final step. Spares the shared FS on
    the hot path."""

    use_tmpdir_sif: bool = False
    """Also copy the SIF to ``$SLURM_TMPDIR`` before ``apptainer exec``.
    Only meaningful when ``use_tmpdir`` is set."""


def render_sbatch_script(spec: SlurmJobSpec) -> str:
    """Return the literal text of the sbatch script for ``spec``.

    The script is deterministic given the spec — golden-file tests pin
    its shape so subtle drift surfaces as a test diff rather than a
    silent change in scheduler behaviour.
    """
    lines: List[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={shlex.quote(spec.job_name)}")
    if spec.partition:
        lines.append(f"#SBATCH --partition={spec.partition}")
    if spec.account:
        lines.append(f"#SBATCH --account={spec.account}")
    if spec.qos:
        lines.append(f"#SBATCH --qos={spec.qos}")
    if spec.time_minutes is not None:
        lines.append(f"#SBATCH --time={int(spec.time_minutes)}")
    if spec.mem_gb is not None:
        lines.append(f"#SBATCH --mem={int(spec.mem_gb)}G")
    lines.append(f"#SBATCH --cpus-per-task={int(spec.cpus_per_task)}")
    if spec.gpus is not None and spec.gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{int(spec.gpus)}")
    if spec.output_log is not None:
        lines.append(f"#SBATCH --output={shlex.quote(str(spec.output_log))}")
    if spec.error_log is not None:
        lines.append(f"#SBATCH --error={shlex.quote(str(spec.error_log))}")
    for directive in spec.extra_directives:
        cleaned = directive.strip()
        if not cleaned:
            continue
        if cleaned.startswith("#SBATCH"):
            lines.append(cleaned)
        else:
            lines.append(f"#SBATCH {cleaned}")

    lines.append("")
    lines.append("set -euo pipefail")
    for key, value in sorted(spec.env.items()):
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append("")

    # When staging is enabled, copy the large inputs to node-local scratch
    # ($SLURM_TMPDIR) before execution to avoid hammering the shared
    # filesystem, and run the container against the local copies. The
    # container's /output is bound to a scratch directory; results are copied
    # back into the workspace as the final step so the promotion saga (which
    # reads the workspace after sacct reports COMPLETED) sees them.
    if spec.use_tmpdir:
        lines.append(
            "cp " + shlex.quote(str(spec.dataset_path)) + ' "$SLURM_TMPDIR/data.h5mu"'
        )
        if spec.use_tmpdir_sif:
            lines.append(
                "cp " + shlex.quote(str(spec.image_sif)) + ' "$SLURM_TMPDIR/image.sif"'
            )
        lines.append('mkdir -p "$SLURM_TMPDIR/output"')
        # The orchestrator wrote job_spec.json into the workspace before
        # sbatch. In tmpdir mode /output is bound to scratch, not the
        # workspace, so stage the spec into scratch before launch — otherwise
        # /output/job_spec.json is missing inside the container. The trailing
        # copy-back step below preserves it in the workspace too.
        job_spec_src = shlex.quote(str(spec.workspace / JOB_SPEC_FILENAME))
        lines.append(f'cp {job_spec_src} "$SLURM_TMPDIR/output/{JOB_SPEC_FILENAME}"')
        lines.append("")

    # Apptainer invocation: bind the dataset (RO) and an output dir (RW) into
    # the container, then run the user command.
    if spec.use_tmpdir:
        dataset_bind_src = '"$SLURM_TMPDIR/data.h5mu"'
        output_bind_src = '"$SLURM_TMPDIR/output"'
    else:
        dataset_bind_src = shlex.quote(str(spec.dataset_path))
        output_bind_src = shlex.quote(str(spec.workspace))

    if spec.use_tmpdir_sif:
        sif_path = '"$SLURM_TMPDIR/image.sif"'
    else:
        sif_path = shlex.quote(str(spec.image_sif))

    bind_args = [
        f"--bind {dataset_bind_src}:/input/data.h5mu:ro",
        f"--bind {output_bind_src}:/output:rw",
    ]
    command = " ".join(shlex.quote(part) for part in spec.command)
    lines.append(
        f"{spec.apptainer_bin} exec {' '.join(bind_args)} "
        f"{sif_path} {command}".rstrip()
    )

    if spec.use_tmpdir:
        # Copy scratch outputs into the workspace. The trailing "/." copies the
        # directory *contents* so files land directly under the workspace,
        # matching where the non-staged path writes them. set -euo pipefail
        # (emitted above) makes a partial copy fail the job rather than promote
        # an incomplete bundle.
        workspace_q = shlex.quote(str(spec.workspace))
        lines.append(f"mkdir -p {workspace_q}")
        lines.append(f'cp -r "$SLURM_TMPDIR/output/." {workspace_q}/')

    lines.append("")
    return "\n".join(lines)
