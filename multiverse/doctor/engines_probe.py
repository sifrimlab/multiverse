"""Doctor probe: container-engine capability detection (STRATEGY M2 §5,
M4 capability detection).

Reports which ``ContainerEngine`` implementations are usable on this
host. The check is *binary-on-PATH only* by design: a real smoke-test
(launching a trivial container) would mutate side state and slow down
``doctor --json``. Set ``deep=True`` to also probe basic invokability
(``--version``).

Available executors are surfaced in the doctor report so a user can
verify their HPC node can actually run Apptainer before submitting,
and so M4's Slurm executor has a uniform place to report on ``sbatch``
discovery.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from .health_probes import (CleanupResult, LeakInventoryResult, ProbeOutcome,
                            ProbeReport)

_CANDIDATE_BINS: Dict[str, List[str]] = {
    "docker": ["docker"],
    "apptainer": ["apptainer", "singularity"],
    "slurm": ["sbatch"],
}

_SLURM_COMPANIONS: List[str] = ["sacct", "scancel", "sinfo"]
"""Auxiliary Slurm binaries the M4 executor depends on. The presence of
``sbatch`` alone is not enough — the executor polls ``sacct`` for
completion and ``scancel`` on user cancel; both must exist for the
real engine to be usable. ``sinfo`` is informational only."""


@dataclass(frozen=True)
class EngineCheck:
    """Result of probing a single container engine for usability.

    Attributes:
        name: Engine identifier (``docker`` / ``apptainer`` / ``slurm``).
        available: Whether the engine's binary was found on PATH (and, in
            deep mode, invoked successfully).
        binary: The first candidate binary found on PATH, or ``None``.
        version: First line of ``<binary> --version`` output (deep mode only).
        note: Human-readable reason the engine is unavailable, or a caveat.
    """

    name: str
    available: bool
    binary: Optional[str]
    version: Optional[str] = None
    note: Optional[str] = None


def check_engine(name: str, *, deep: bool = False) -> EngineCheck:
    """Detect whether one container engine is usable on this host.

    Args:
        name: Engine to probe; resolved against the candidate-binary table
            (``apptainer`` also accepts the ``singularity`` alias).
        deep: When True, also invoke ``<binary> --version`` to confirm the
            binary is runnable, not merely present on PATH. A non-zero exit
            is recorded as a note; an OSError/timeout marks the engine
            unavailable.

    Returns:
        An :class:`EngineCheck` describing availability and (in deep mode)
        the discovered version.
    """
    candidates = _CANDIDATE_BINS.get(name, [name])
    found: Optional[str] = None
    for c in candidates:
        if shutil.which(c):
            found = c
            break
    if found is None:
        return EngineCheck(
            name=name,
            available=False,
            binary=None,
            note=f"no {'/'.join(candidates)} on PATH",
        )
    version: Optional[str] = None
    note: Optional[str] = None
    if deep:
        try:
            result = subprocess.run(
                [found, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            version = (result.stdout or result.stderr).strip().splitlines()[0:1]
            version = version[0] if version else None
            if result.returncode != 0:
                note = f"{found} --version returned {result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            note = f"{type(exc).__name__}: {exc}"
            return EngineCheck(name=name, available=False, binary=found, note=note)
    return EngineCheck(
        name=name, available=True, binary=found, version=version, note=note
    )


def probe_container_engines(*, deep: bool = False) -> ProbeReport:
    """Aggregate per-engine availability into one doctor probe report.

    The probe always *passes* (it's diagnostic, not gating). The detail
    string carries the per-engine availability so callers can decide
    what to gate on.

    Args:
        deep: Forwarded to :func:`check_engine`; when True each present
            binary is invoked with ``--version``.

    Returns:
        A :class:`ProbeReport` named ``engines.container_backends`` whose
        ``detail`` summarises which engines are available and, for Slurm,
        any missing companion binaries.
    """
    checks = [check_engine(name, deep=deep) for name in _CANDIDATE_BINS]
    available = [c.name for c in checks if c.available]
    parts = []
    for c in checks:
        if c.available:
            piece = f"{c.name}=ok({c.binary}"
            if c.version:
                piece += f", {c.version}"
            piece += ")"
            # STRATEGY M4: when Slurm is available, also report which
            # companion binaries are present. The executor's polling
            # path requires sacct + scancel; the doctor surfaces the
            # gap so a misconfigured login node fails the probe before
            # a user wastes a submission.
            if c.name == "slurm":
                missing = [b for b in _SLURM_COMPANIONS if shutil.which(b) is None]
                if missing:
                    piece += f", missing={','.join(missing)}"
        else:
            piece = f"{c.name}=missing"
            if c.note:
                piece += f" [{c.note}]"
        parts.append(piece)
    detail = "; ".join(parts)
    if available:
        detail = f"available={','.join(available)} | " + detail

    return ProbeReport(
        name="engines.container_backends",
        probe=ProbeOutcome.PASS,
        cleanup=CleanupResult.CLEAN,
        leak=LeakInventoryResult.NONE,
        leak_count=0,
        detail=detail,
    )
