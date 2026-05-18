"""Typed session-state helpers for the Streamlit GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st


@dataclass(frozen=True)
class GuiState:
    selected_datasets: list[str]
    selected_models: list[str]
    planned_jobs: list[dict]
    run_mode: Literal["Use User Params", "Run Gridsearch"]
    registry_dirty: bool
    editor_version: int
    experiment_name: str
    active_experiment_id: str | None
    active_experiment_name: str
    is_running: bool
    pending_launch: dict | None


STATE_DEFAULTS: dict = {
    "selected_datasets": [],
    "selected_models": [],
    "planned_jobs": [],
    "run_mode": "Use User Params",
    "registry_dirty": False,
    "editor_version": 0,
    "experiment_name": "benchmark_run",
    "active_experiment_id": None,
    "active_experiment_name": "",
    "is_running": False,
    "pending_launch": None,
}


def init_state() -> GuiState:
    for key, default in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)
    return get_state()


def get_state() -> GuiState:
    state = st.session_state
    run_mode = state.get("run_mode", STATE_DEFAULTS["run_mode"])
    if run_mode not in {"Use User Params", "Run Gridsearch"}:
        run_mode = STATE_DEFAULTS["run_mode"]
        state["run_mode"] = run_mode

    return GuiState(
        selected_datasets=list(state.get("selected_datasets", [])),
        selected_models=list(state.get("selected_models", [])),
        planned_jobs=list(state.get("planned_jobs", [])),
        run_mode=run_mode,
        registry_dirty=bool(state.get("registry_dirty", False)),
        editor_version=int(state.get("editor_version", 0) or 0),
        experiment_name=str(state.get("experiment_name", "benchmark_run") or "benchmark_run"),
        active_experiment_id=state.get("active_experiment_id"),
        active_experiment_name=str(state.get("active_experiment_name", "")),
        is_running=bool(state.get("is_running", False)),
        pending_launch=state.get("pending_launch"),
    )


def bump_editor_version() -> int:
    st.session_state["editor_version"] = int(st.session_state.get("editor_version", 0) or 0) + 1
    return int(st.session_state["editor_version"])
