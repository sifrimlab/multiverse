"""Query-param navigation helpers for the Streamlit GUI."""

from __future__ import annotations

import streamlit as st

TAB_LABELS: dict[str, str] = {
    "registry": "Registry",
    "configure": "Configure",
    "run": "Run",
    "results": "Results",
    "analysis": "Analysis",
}
TABS = list(TAB_LABELS.keys())
LEGACY_TAB_REDIRECTS: dict[str, str] = {
    "jobs": "configure",
    "params": "configure",
    "execute": "run",
    "mlflow": "analysis",
    "optuna": "analysis",
    "settings": "registry",
}


def _query_tab() -> str:
    """Resolve the active tab slug from the ``tab`` query parameter.

    Folds legacy slugs to their current equivalents (rewriting the URL) and
    falls back to ``registry`` for unknown values.
    """
    tab = st.query_params.get("tab", "registry")
    if isinstance(tab, list):
        tab = tab[0] if tab else "registry"
    if tab in LEGACY_TAB_REDIRECTS:
        tab = LEGACY_TAB_REDIRECTS[tab]
        st.query_params["tab"] = tab
    return tab if tab in TABS else "registry"


def go_to(tab_slug: str) -> None:
    """Navigate to a tab by writing the query param and rerunning the app.

    Args:
        tab_slug: Target tab slug; legacy slugs are mapped forward, and unknown
            slugs leave the current tab unchanged before the rerun.
    """
    tab_slug = LEGACY_TAB_REDIRECTS.get(tab_slug, tab_slug)
    if tab_slug in TABS:
        st.query_params["tab"] = tab_slug
    st.rerun()


def current_tab_slug() -> str:
    """Return the slug of the currently active tab."""
    return _query_tab()


def render_top_nav() -> str:
    """Render the top navigation bar and return the active tab slug.

    Draws one button per tab, highlighting the active one; clicking a different
    tab updates the query param and reruns.

    Returns:
        The active tab slug as resolved at render time.
    """
    current = current_tab_slug()
    cols = st.columns(len(TABS))
    for col, slug in zip(cols, TABS):
        label = TAB_LABELS[slug]
        button_type = "primary" if slug == current else "secondary"
        with col:
            if st.button(
                label, key=f"top_nav_{slug}", type=button_type, width="stretch"
            ):
                if slug != current:
                    st.query_params["tab"] = slug
                    st.rerun()
    return current


def render_workflow_stepper() -> None:
    """Deprecated compatibility shim; top navigation is now canonical."""
    return None
