"""Apply a GC plan or write a dry-run report.

``apply_plan`` is the *only* code path in the codebase that calls
``shutil.rmtree`` against user-visible directories. The corresponding
test in ``tests/unit/test_gc.py::test_dry_run_default_never_deletes``
locks this invariant.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..artifact.checksums import atomic_write_bytes
from .errors import GcGateError, NotOwnedError
from .plan import GcPlan, PlanEntry, PlanReason

GC_REPORTS_SUBDIR = "gc_reports"


@dataclass
class GcResult:
    plan: GcPlan
    deleted_paths: List[Path] = field(default_factory=list)
    refused_paths: List[Path] = field(default_factory=list)
    report_path: Optional[Path] = None


def apply_plan(
    plan: GcPlan,
    *,
    store_root: Path,
    apply: bool,
) -> GcResult:
    """Either write the dry-run report (``apply=False``, default) or
    perform deletions (``apply=True``).

    Even with ``apply=True``, every entry still goes through the gate
    re-check (owner token present); a race that flipped a directory's
    state since the plan was built is refused, not skipped.
    """
    result = GcResult(plan=plan)
    if not apply:
        result.report_path = write_dry_run_report(plan, store_root=store_root)
        return result

    for entry in plan.to_delete:
        try:
            _do_delete(entry)
            result.deleted_paths.append(entry.candidate.path)
        except GcGateError:
            result.refused_paths.append(entry.candidate.path)
            continue

    result.report_path = write_dry_run_report(
        plan, store_root=store_root, header="GC apply report"
    )
    return result


def _do_delete(entry: PlanEntry) -> None:
    """Final pre-delete re-check + rmtree.

    The token is re-read at this moment to defend against a race where the
    plan saw a token but it has since been removed.
    """
    candidate = entry.candidate
    if entry.reason is not PlanReason.WOULD_DELETE:
        raise GcGateError(f"plan entry {candidate.path} is not marked for deletion")
    from ..promotion.tokens import read_owner_token

    token = read_owner_token(candidate.path)
    if token is None:
        raise NotOwnedError(
            f"refusing to delete {candidate.path}: .mvd_owner vanished between plan and apply"
        )
    shutil.rmtree(candidate.path)


def write_dry_run_report(
    plan: GcPlan, *, store_root: Path, header: str = "GC dry-run report"
) -> Path:
    """Write ``store/gc_reports/<rfc3339>.md``. Returns the report path."""
    reports = store_root / GC_REPORTS_SUBDIR
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ%f")
    path = reports / f"{stamp}.md"

    lines = [
        f"# {header}",
        f"_generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Would delete",
    ]
    if not plan.to_delete:
        lines.append("_nothing_")
    else:
        for entry in plan.to_delete:
            lines.append(f"- `{entry.candidate.path}` ({entry.candidate.kind.value})")
    lines += ["", "## Kept"]
    if not plan.to_keep:
        lines.append("_nothing_")
    else:
        for entry in plan.to_keep:
            note = f" — {entry.note}" if entry.note else ""
            lines.append(
                f"- `{entry.candidate.path}` ({entry.candidate.kind.value}): "
                f"{entry.reason.value}{note}"
            )
    lines.append("")
    atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))
    return path
