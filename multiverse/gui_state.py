"""Typed session-state helpers for the Streamlit GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st


@dataclass(frozen=True)
class GuiState:
    """Immutable snapshot of the GUI's session state.

    A read-only projection of ``st.session_state`` produced by ``get_state``;
    it is not the source of truth for any run, only a view the render code
    consumes.

    Attributes:
        selected_datasets: Dataset slugs the user has selected in the registry.
        selected_models: Model slugs the user has selected in the registry.
        planned_jobs: Planned launch members (one dict per dataset/model job).
        run_mode: Effective execution mode (fixed user params vs. gridsearch).
        registry_dirty: Whether registry selections have unsaved edits.
        editor_version: Monotonic counter bumped to force editor widget resets.
        shared_experiment_name: Experiment name shared across tabs.
        shared_seed: Random seed shared across tabs.
        shared_run_mode: Run mode shared across tabs (canonical source).
        shared_manifest_path: Path to the run manifest shared across tabs.
        experiment_name: Mirror of ``shared_experiment_name`` for legacy reads.
        active_experiment_id: MLflow experiment id of the in-flight launch, if any.
        active_experiment_name: MLflow experiment name of the in-flight launch.
        is_running: Whether a launch is currently executing.
        pending_launch: Launch request awaiting confirmation/start, if any.
    """

    selected_datasets: list[str]
    selected_models: list[str]
    planned_jobs: list[dict]
    run_mode: Literal["Use User Params", "Run Gridsearch"]
    registry_dirty: bool
    editor_version: int
    shared_experiment_name: str
    shared_seed: int
    shared_run_mode: Literal["Use User Params", "Run Gridsearch"]
    shared_manifest_path: str
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
    "shared_run_mode": "Use User Params",
    "registry_dirty": False,
    "editor_version": 0,
    "experiment_name": "benchmark_run",
    "shared_experiment_name": "benchmark_run",
    "shared_seed": 42,
    "shared_manifest_path": "run_manifest.yaml",
    "active_experiment_id": None,
    "active_experiment_name": "",
    "is_running": False,
    "pending_launch": None,
}


def _migrate_shared_state() -> None:
    """Backfill shared-state keys from older per-tab keys, once per session."""
    state = st.session_state
    if not state.get("_shared_state_migrated"):
        if "shared_experiment_name" not in state and "experiment_name" in state:
            state["shared_experiment_name"] = state["experiment_name"]
        if "shared_seed" not in state:
            if "jb_seed" in state:
                state["shared_seed"] = state["jb_seed"]
            elif "exec_seed" in state:
                state["shared_seed"] = state["exec_seed"]
        if "shared_manifest_path" not in state:
            if "jb_manifest_path" in state:
                state["shared_manifest_path"] = state["jb_manifest_path"]
            elif "exec_manifest_path" in state:
                state["shared_manifest_path"] = state["exec_manifest_path"]
        if "shared_run_mode" not in state and "run_mode" in state:
            state["shared_run_mode"] = state["run_mode"]
        state["_shared_state_migrated"] = True


def _apply_pending_shared_state() -> None:
    """Commit deferred shared-state writes staged on the previous run.

    Widgets cannot mutate their own bound session key during the run that reads
    it, so updates are parked under ``_pending_*`` keys and applied here at the
    start of the next run.
    """
    state = st.session_state
    pending_map = {
        "_pending_shared_experiment_name": "shared_experiment_name",
        "_pending_shared_seed": "shared_seed",
        "_pending_shared_run_mode": "shared_run_mode",
        "_pending_shared_manifest_path": "shared_manifest_path",
    }
    for pending_key, target_key in pending_map.items():
        if pending_key in state:
            state[target_key] = state.pop(pending_key)


def init_state() -> GuiState:
    """Initialise session state and return its current snapshot.

    Runs the one-time migrations, applies any deferred writes, seeds defaults
    for missing keys, and syncs the legacy mirror keys to their shared sources.

    Returns:
        The current :class:`GuiState` snapshot after initialisation.
    """
    _migrate_shared_state()
    _apply_pending_shared_state()
    for key, default in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)
    st.session_state["experiment_name"] = st.session_state.get(
        "shared_experiment_name", "benchmark_run"
    )
    st.session_state["run_mode"] = st.session_state.get(
        "shared_run_mode", "Use User Params"
    )
    return get_state()


def get_state() -> GuiState:
    """Build a validated :class:`GuiState` snapshot from session state.

    Coerces and defaults each field defensively (invalid run modes, missing or
    falsy values) so callers always receive a well-formed snapshot.

    Returns:
        The current :class:`GuiState` snapshot.
    """
    state = st.session_state
    run_mode = state.get(
        "shared_run_mode", state.get("run_mode", STATE_DEFAULTS["run_mode"])
    )
    if run_mode not in {"Use User Params", "Run Gridsearch"}:
        run_mode = STATE_DEFAULTS["shared_run_mode"]

    experiment_name = str(
        state.get(
            "shared_experiment_name", state.get("experiment_name", "benchmark_run")
        )
        or "benchmark_run"
    )
    seed = int(state.get("shared_seed", 42) or 42)
    manifest_path = str(
        state.get("shared_manifest_path", "run_manifest.yaml") or "run_manifest.yaml"
    )

    return GuiState(
        selected_datasets=list(state.get("selected_datasets", [])),
        selected_models=list(state.get("selected_models", [])),
        planned_jobs=list(state.get("planned_jobs", [])),
        run_mode=run_mode,
        registry_dirty=bool(state.get("registry_dirty", False)),
        editor_version=int(state.get("editor_version", 0) or 0),
        shared_experiment_name=experiment_name,
        shared_seed=seed,
        shared_run_mode=run_mode,
        shared_manifest_path=manifest_path,
        experiment_name=experiment_name,
        active_experiment_id=state.get("active_experiment_id"),
        active_experiment_name=str(state.get("active_experiment_name", "")),
        is_running=bool(state.get("is_running", False)),
        pending_launch=state.get("pending_launch"),
    )


def bump_editor_version() -> int:
    """Increment the editor version counter to force widget remounts.

    Returns:
        The new editor version value.
    """
    st.session_state["editor_version"] = (
        int(st.session_state.get("editor_version", 0) or 0) + 1
    )
    return int(st.session_state["editor_version"])
