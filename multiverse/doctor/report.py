"""Doctor report assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SectionStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class DoctorSection:
    name: str
    status: SectionStatus
    rows: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "rows": [dict(r) for r in self.rows],
            "summary": self.summary,
        }


@dataclass
class DoctorReport:
    generated_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).astimezone().isoformat()
    )
    sections: List[DoctorSection] = field(default_factory=list)
    accept_degraded: bool = False

    @property
    def overall_status(self) -> SectionStatus:
        statuses = [s.status for s in self.sections]
        if SectionStatus.BLOCKED in statuses:
            return SectionStatus.BLOCKED
        if SectionStatus.WARNING in statuses:
            return SectionStatus.WARNING
        return SectionStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_iso": self.generated_iso,
            "overall_status": self.overall_status.value,
            "accept_degraded": self.accept_degraded,
            "sections": [s.to_dict() for s in self.sections],
        }
