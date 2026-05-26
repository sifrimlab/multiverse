"""Query-param navigation helpers for the Streamlit GUI."""

from __future__ import annotations

import streamlit as st


TAB_LABELS: dict[str, str] = {
    "registry": "Registry",
    "jobs": "Job Builder",
    "params": "Parameters",
    "execute": "Execute",
    "results": "Results",
    "mlflow": "Experiment Analysis",
    "optuna": "Sweep Tracker",
    "settings": "Settings",
}
TABS = list(TAB_LABELS.keys())


def _query_tab() -> str:
    tab = st.query_params.get("tab", "registry")
    if isinstance(tab, list):
        tab = tab[0] if tab else "registry"
    return tab if tab in TABS else "registry"


def go_to(tab_slug: str) -> None:
    if tab_slug in TABS:
        st.query_params["tab"] = tab_slug
    st.rerun()


def current_tab_slug() -> str:
    return _query_tab()


def render_top_nav() -> str:
    current = current_tab_slug()
    labels = [TAB_LABELS[slug] for slug in TABS]
    selected = st.radio(
        "Section",
        options=labels,
        index=TABS.index(current),
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_slug = TABS[labels.index(selected)]
    if selected_slug != current:
        st.query_params["tab"] = selected_slug
        st.rerun()
    return selected_slug


def render_workflow_stepper() -> None:
    st.subheader("Workflow")
    current = current_tab_slug()
    for idx, slug in enumerate(TABS, start=1):
        label = f"{idx}. {TAB_LABELS[slug]}"
        if slug == current:
            st.caption(f"Current: {label}")
        elif st.button(label, key=f"nav_step_{slug}"):
            go_to(slug)
