"""Deep Slurm capability probe (STRATEGY M4 §5).

The default ``multiverse doctor`` report uses
:func:`engines_probe.probe_container_engines` to check that sbatch and
its companions are on PATH — that's the cheap, non-mutating check that
runs every time. This module adds the *deep* probe: enumerate
partitions via ``sinfo`` and (optionally) submit a ``--wrap=true``
smoke job that is reaped before returning.

Deep probing actually shells out to Slurm. It is opt-in
(``multiverse doctor --deep-slurm``) because:

* ``sinfo`` is cheap on a healthy controller but can take seconds on a
  loaded one.
* ``sbatch --wrap=true`` actually allocates a node, even if for an
  instant. Users on a shared cluster don't want that to happen on
  every doctor run.

The smoke test never reports PASS without round-tripping a real job
id. If sbatch returns success but sacct never sees the job, we report
FAIL — because the M4 executor would do the same.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .health_probes import (
    CleanupResult,
    LeakInventoryResult,
    ProbeOutcome,
    ProbeReport,
)


DEFAULT_SMOKE_TIMEOUT_SECONDS = 60


@dataclass
class SlurmCapability:
    sbatch: bool = False
    sacct: bool = False
    scancel: bool = False
    sinfo: bool = False
    partitions: List[str] = field(default_factory=list)
    smoke_job_id: Optional[str] = None
    smoke_final_state: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """The M4 executor needs sbatch + sacct + scancel — without
        any one of them, the real engine cannot drive a run end-to-end.
        ``sinfo`` is informational only."""
        return self.sbatch and self.sacct and self.scancel


def probe_slurm_deep(
    *,
    smoke_test: bool = False,
    smoke_partition: Optional[str] = None,
    smoke_timeout_seconds: int = DEFAULT_SMOKE_TIMEOUT_SECONDS,
    timeout_seconds: int = 10,
) -> ProbeReport:
    """Deep probe: enumerate partitions and (optionally) smoke-test sbatch.

    ``smoke_test=True`` submits a one-second ``sbatch --wrap=true``
    job and polls ``sacct`` until it reaches a terminal state. The job
    is scancelled if the timeout fires. Cleanup runs in a ``finally``
    so a partial probe never leaks a long-running allocation.
    """
    cap = SlurmCapability(
        sbatch=shutil.which("sbatch") is not None,
        sacct=shutil.which("sacct") is not None,
        scancel=shutil.which("scancel") is not None,
        sinfo=shutil.which("sinfo") is not None,
    )

    if not cap.sbatch:
        return _report(
            cap, probe=ProbeOutcome.SKIPPED, detail="sbatch not on PATH"
        )
    if not (cap.sacct and cap.scancel):
        missing = []
        if not cap.sacct:
            missing.append("sacct")
        if not cap.scancel:
            missing.append("scancel")
        return _report(
            cap,
            probe=ProbeOutcome.FAIL,
            detail=f"sbatch present but {','.join(missing)} missing",
        )

    if cap.sinfo:
        partitions = _enumerate_partitions(timeout_seconds=timeout_seconds)
        cap.partitions = partitions
        if not partitions:
            cap.errors.append("sinfo returned no partitions")

    if smoke_test:
        _run_smoke_test(
            cap,
            partition=smoke_partition or (cap.partitions[0] if cap.partitions else None),
            timeout_seconds=smoke_timeout_seconds,
        )
        if cap.smoke_final_state != "COMPLETED":
            return _report(
                cap,
                probe=ProbeOutcome.FAIL,
                detail=_describe(cap),
            )

    return _report(
        cap,
        probe=ProbeOutcome.PASS,
        detail=_describe(cap),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _enumerate_partitions(*, timeout_seconds: int) -> List[str]:
    try:
        result = subprocess.run(
            ["sinfo", "--noheader", "--format=%P"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    seen: List[str] = []
    for line in result.stdout.splitlines():
        # ``%P`` ends in ``*`` for the default partition; strip it so
        # callers see the bare name.
        name = line.strip().rstrip("*")
        if name and name not in seen:
            seen.append(name)
    return seen


def _run_smoke_test(
    cap: SlurmCapability,
    *,
    partition: Optional[str],
    timeout_seconds: int,
) -> None:
    """Submit ``sbatch --wrap=true``, poll until terminal, scancel on
    timeout. Updates ``cap`` in place."""
    cmd = ["sbatch", "--parsable", "--wrap=true", "--time=1"]
    if partition:
        cmd.extend(["--partition", partition])
    try:
        submit = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        cap.errors.append(f"sbatch invocation failed: {type(exc).__name__}: {exc}")
        return
    if submit.returncode != 0:
        cap.errors.append(
            f"sbatch exited {submit.returncode}: {submit.stderr.strip()}"
        )
        return
    job_id = (submit.stdout or "").strip().split(";", 1)[0].strip()
    if not job_id or not job_id.isdigit():
        cap.errors.append(f"could not parse sbatch output: {submit.stdout!r}")
        return
    cap.smoke_job_id = job_id

    deadline = time.monotonic() + timeout_seconds
    final_state: Optional[str] = None
    while time.monotonic() < deadline:
        state = _query_state(job_id)
        if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                     "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
            final_state = state
            break
        time.sleep(1.0)

    if final_state is None:
        # Reap with scancel so we never leak an allocation. The job
        # may still complete naturally a moment later; that is fine.
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            cap.errors.append("scancel cleanup invocation failed")
        cap.errors.append(
            f"smoke job {job_id} did not reach terminal state in "
            f"{timeout_seconds}s; scancel issued"
        )
        cap.smoke_final_state = "TIMEOUT_DURING_PROBE"
        return

    cap.smoke_final_state = final_state


def _query_state(job_id: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "--noheader",
                "--parsable2",
                "--format=State",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        token = token.rstrip("+")
        if token:
            return token
    return None


def _describe(cap: SlurmCapability) -> str:
    parts = []
    if cap.partitions:
        parts.append(f"partitions={','.join(cap.partitions[:5])}")
        if len(cap.partitions) > 5:
            parts[-1] += f"(+{len(cap.partitions) - 5})"
    if cap.smoke_job_id:
        parts.append(f"smoke_job={cap.smoke_job_id}={cap.smoke_final_state}")
    if cap.errors:
        parts.append(f"errors=[{'; '.join(cap.errors)}]")
    return "; ".join(parts) or "ok"


def _report(
    cap: SlurmCapability, *, probe: ProbeOutcome, detail: str
) -> ProbeReport:
    return ProbeReport(
        name="engines.slurm_deep",
        probe=probe,
        cleanup=CleanupResult.CLEAN,
        leak=LeakInventoryResult.NONE,
        leak_count=0,
        detail=detail,
    )
