"""Doctor report assembly: sections, statuses, and JSON projection.

Holds the data model the ``multiverse doctor`` command renders. A report is
a list of sections; the overall status is the worst section status. The
model is purely a read surface — assembling it never mutates state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SectionStatus(str, Enum):
    """Worst-of severity for a section: ``BLOCKED`` outranks ``WARNING``
    outranks ``OK``."""

    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class DoctorSection:
    """One named group of probe rows with a rolled-up status.

    Attributes:
        name: Section heading shown in the report.
        status: Section-level severity.
        rows: Per-probe rows (e.g. ``ProbeReport.to_dict()`` outputs).
        summary: Optional one-line summary of the section.
    """

    name: str
    status: SectionStatus
    rows: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Render the section as a JSON-serialisable dict."""
        return {
            "name": self.name,
            "status": self.status.value,
            "rows": [dict(r) for r in self.rows],
            "summary": self.summary,
        }


@dataclass
class DoctorReport:
    """Top-level doctor report: a timestamped collection of sections.

    Attributes:
        generated_iso: Local-timezone ISO-8601 timestamp of assembly.
        sections: The report's sections, in display order.
        accept_degraded: Whether the run was invoked with
            ``--accept-degraded`` (recorded so the JSON output is
            self-describing).
    """

    generated_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).astimezone().isoformat()
    )
    sections: List[DoctorSection] = field(default_factory=list)
    accept_degraded: bool = False

    @property
    def overall_status(self) -> SectionStatus:
        """Worst status across all sections (``OK`` when empty)."""
        statuses = [s.status for s in self.sections]
        if SectionStatus.BLOCKED in statuses:
            return SectionStatus.BLOCKED
        if SectionStatus.WARNING in statuses:
            return SectionStatus.WARNING
        return SectionStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        """Render the whole report as the ``doctor --json`` payload."""
        return {
            "generated_iso": self.generated_iso,
            "overall_status": self.overall_status.value,
            "accept_degraded": self.accept_degraded,
            "sections": [s.to_dict() for s in self.sections],
        }
