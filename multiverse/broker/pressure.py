"""Pressure modes (STRATEGY R11)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PressureMode(str, Enum):
    NORMAL = "normal"
    """Admit, run, no intervention."""

    PRESSURE = "pressure"
    """Stop admitting new jobs. Running jobs continue untouched."""

    CRITICAL = "critical"
    """Stop admitting; emit a warning event. Do NOT auto-kill running jobs."""


@dataclass(frozen=True)
class PressureThresholds:
    """Fraction-of-host thresholds for transitions between modes.

    Defaults derived from common local-workstation behaviour:
    PRESSURE at 85% utilisation, CRITICAL at 95%. The strategy explicitly
    rejects automatic preemption (R11), so CRITICAL only stops admissions.
    """

    ram_pressure: float = 0.85
    ram_critical: float = 0.95
    vram_pressure: float = 0.85
    vram_critical: float = 0.95
    disk_pressure: float = 0.90
    disk_critical: float = 0.98
