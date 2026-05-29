"""Slurm engine protocol + real subprocess adapter (STRATEGY M4).

The kernel's executor talks to *some* engine to submit, poll, and cancel
Slurm jobs. The protocol is intentionally narrow: ``submit / query /
cancel`` and a ``name`` for telemetry. Production wires
:class:`RealSlurmEngine` (shells out to ``sbatch`` / ``sacct`` /
``scancel``); tests wire :class:`~multiverse.slurm.fake.InMemorySlurmEngine`.

The real engine does *not* implement retries or rate-limiting itself —
those belong to the broker (M4 §2). It is a thin wrapper that fails
loudly when Slurm misbehaves.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

from ..apptainer.images import compute_sif_digest
from .state import SlurmJobInfo, SlurmJobState, from_sacct_state
from .template import SlurmJobSpec, render_sbatch_script


class SlurmEngineError(RuntimeError):
    """Raised when the underlying Slurm CLI misbehaves in a way the
    executor cannot recover from (missing binary, unparseable output,
    nonzero exit on submit). The executor catches this and transitions
    the run to ``FAILED``."""


@dataclass(frozen=True)
class SlurmSubmission:
    job_id: str
    script_path: Path


@runtime_checkable
class SlurmEngine(Protocol):
    """Narrow protocol for everything the executor needs from Slurm."""

    name: str

    def submit(self, spec: SlurmJobSpec, *, script_dir: Path) -> SlurmSubmission: ...

    def query(self, job_id: str) -> SlurmJobInfo: ...

    def cancel(self, job_id: str) -> None: ...

    def sif_digest_for_submission(self, spec: SlurmJobSpec) -> Optional[str]:
        """Return the sha256 digest of the SIF at ``spec.image_sif``, or
        ``None`` if the engine cannot or does not compute a digest.

        The digest is ``sha256:<hex>``. Engines that implement this hook
        enable the M2 dual-digest invariant for Slurm manifests.
        """
        ...


# ---------------------------------------------------------------------------
# Real engine: subprocess wrapper around sbatch / sacct / scancel
# ---------------------------------------------------------------------------


_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (?P<job_id>\d+)")


@dataclass
class RealSlurmEngine:
    """Production Slurm engine. Shells out to the local Slurm CLI.

    The engine is intentionally stateless — every ``query`` re-invokes
    ``sacct`` so a kernel restart re-derives state from the scheduler,
    not from a local cache. ``submit`` also writes the rendered script
    next to the workspace for forensics; the path is returned in the
    submission record and recorded in the manifest's lineage block.
    """

    sbatch_bin: str = "sbatch"
    sacct_bin: str = "sacct"
    scancel_bin: str = "scancel"
    timeout_seconds: int = 30
    name: str = "slurm-real"
    # Cache key: (absolute_path, mtime_ns, size_bytes) → sha256:hex
    _sif_digest_cache: Dict[Tuple[str, int, int], str] = field(
        default_factory=dict, init=False, repr=False
    )

    def submit(self, spec: SlurmJobSpec, *, script_dir: Path) -> SlurmSubmission:
        self._require_binary(self.sbatch_bin)
        script_dir = Path(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{spec.job_name}.sbatch"
        script_path.write_text(render_sbatch_script(spec), encoding="utf-8")
        try:
            result = subprocess.run(
                [self.sbatch_bin, "--parsable", str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmEngineError(
                f"sbatch invocation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if result.returncode != 0:
            raise SlurmEngineError(
                f"sbatch exited {result.returncode}: {result.stderr.strip()}"
            )
        job_id = _parse_sbatch_output(result.stdout)
        if job_id is None:
            raise SlurmEngineError(
                f"could not parse sbatch output: {result.stdout!r}"
            )
        return SlurmSubmission(job_id=job_id, script_path=script_path)

    def query(self, job_id: str) -> SlurmJobInfo:
        self._require_binary(self.sacct_bin)
        try:
            result = subprocess.run(
                [
                    self.sacct_bin,
                    "-j",
                    str(job_id),
                    "--noheader",
                    "--parsable2",
                    "--format=JobID,State,ExitCode,Reason",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmEngineError(
                f"sacct invocation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if result.returncode != 0:
            raise SlurmEngineError(
                f"sacct exited {result.returncode}: {result.stderr.strip()}"
            )
        return _parse_sacct_output(job_id, result.stdout)

    def cancel(self, job_id: str) -> None:
        self._require_binary(self.scancel_bin)
        try:
            subprocess.run(
                [self.scancel_bin, str(job_id)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmEngineError(
                f"scancel invocation failed: {type(exc).__name__}: {exc}"
            ) from exc

    def sif_digest_for_submission(self, spec: SlurmJobSpec) -> Optional[str]:
        """Return the sha256 digest of ``spec.image_sif``, cached by
        (path, mtime_ns, size) so repeated submits of the same SIF
        avoid rehashing."""
        sif = Path(spec.image_sif)
        if not sif.is_file():
            return None
        stat = sif.stat()
        key: Tuple[str, int, int] = (str(sif.resolve()), stat.st_mtime_ns, stat.st_size)
        if key not in self._sif_digest_cache:
            self._sif_digest_cache[key] = compute_sif_digest(sif)
        return self._sif_digest_cache[key]

    # ---- internals ---------------------------------------------------

    def _require_binary(self, name: str) -> None:
        if shutil.which(name) is None:
            raise SlurmEngineError(f"{name!r} not on PATH")


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def _parse_sbatch_output(stdout: str) -> Optional[str]:
    """sbatch --parsable prints ``<job_id>[;<cluster>]`` on stdout; the
    non-parsable form prints ``Submitted batch job <job_id>``. Accept
    both shapes so a misconfigured installation still works."""
    stripped = (stdout or "").strip()
    if not stripped:
        return None
    if stripped[0].isdigit():
        # --parsable: "12345" or "12345;cluster"
        return stripped.split(";", 1)[0].strip()
    m = _SBATCH_JOB_ID_RE.search(stripped)
    return m.group("job_id") if m else None


def _parse_sacct_output(job_id: str, stdout: str) -> SlurmJobInfo:
    """sacct emits one row per step plus a parent row; we want the
    parent (``<job_id>``, without the ``.batch`` / ``.external``
    suffix). If no parent row appears the job is too new to be
    materialized in the accounting database — return ``PENDING``.
    """
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        row_job_id, state_raw = parts[0], parts[1]
        if row_job_id != str(job_id):
            continue
        exit_code = _parse_exit_code(parts[2] if len(parts) >= 3 else "")
        reason = parts[3].strip() if len(parts) >= 4 else None
        return SlurmJobInfo(
            job_id=str(job_id),
            state=from_sacct_state(state_raw),
            exit_code=exit_code,
            reason=reason or None,
        )
    return SlurmJobInfo(
        job_id=str(job_id), state=SlurmJobState.PENDING, exit_code=None
    )


def _parse_exit_code(raw: str) -> Optional[int]:
    """sacct emits exit codes as ``<rc>:<signal>``. We take the rc; a
    nonzero signal with rc=0 is still a failure but the state column
    already captured that signal (e.g. ``OUT_OF_MEMORY``)."""
    token = (raw or "").strip()
    if not token:
        return None
    head = token.split(":", 1)[0].strip()
    if not head:
        return None
    try:
        return int(head)
    except ValueError:
        return None
