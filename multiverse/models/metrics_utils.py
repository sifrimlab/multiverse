from __future__ import annotations

from typing import Any, Dict, List


def series_to_float_list(values: Any) -> List[float]:
    """Convert training history (Series, ndarray, list) to a list of floats."""
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return []
    out: List[float] = []
    for item in values:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def scvi_history_to_dict(history: Any) -> Dict[str, List[float]]:
    """Extract per-epoch metrics from an scvi-tools training history object."""
    if not history:
        return {}
    result: Dict[str, List[float]] = {}
    keys = history.keys() if hasattr(history, "keys") else []
    for key in keys:
        series = series_to_float_list(history[key])
        if series:
            result[str(key)] = series
    return result
