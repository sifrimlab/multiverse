"""Streamlit widget utilities for dynamic hyperparameter forms (T3.1)."""

from __future__ import annotations

import streamlit as st


def _safe_float(v, fallback: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(fallback)


def _safe_int(v, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(fallback)


def _render_fixed_widget(key_prefix: str, param_name: str, spec: dict):
    """Render a single fixed-value widget from a JSON Schema property spec."""
    label = param_name
    default = spec.get("default")
    description = spec.get("description")
    enum_values = spec.get("enum")
    param_type = spec.get("type")
    widget_key = f"{key_prefix}::fixed::{param_name}"

    if isinstance(enum_values, list) and enum_values:
        idx = enum_values.index(default) if default in enum_values else 0
        return st.selectbox(label, options=enum_values, index=idx, key=widget_key, help=description)

    if param_type == "integer":
        schema_min = spec.get("minimum")
        schema_max = spec.get("maximum")
        step = max(1, _safe_int(spec.get("multipleOf", 1), 1))
        return st.number_input(
            label,
            min_value=_safe_int(schema_min) if schema_min is not None else None,
            max_value=_safe_int(schema_max) if schema_max is not None else None,
            value=_safe_int(default, 0),
            step=step,
            key=widget_key,
            help=description,
        )

    if param_type == "number":
        schema_min = spec.get("minimum") if spec.get("minimum") is not None else spec.get("exclusiveMinimum")
        schema_max = spec.get("maximum") if spec.get("maximum") is not None else spec.get("exclusiveMaximum")
        step = _safe_float(spec.get("multipleOf", 0.001), 0.001)
        if step <= 0:
            step = 0.001
        return st.number_input(
            label,
            min_value=_safe_float(schema_min) if schema_min is not None else None,
            max_value=_safe_float(schema_max) if schema_max is not None else None,
            value=_safe_float(default, 0.0),
            step=step,
            key=widget_key,
            format="%.6f",
            help=description,
        )

    if param_type == "boolean":
        return st.checkbox(
            label,
            value=bool(default) if default is not None else False,
            key=widget_key,
            help=description,
        )

    return st.text_input(
        label,
        value="" if default is None else str(default),
        key=widget_key,
        help=description,
    )


def _render_sweep_widget(key_prefix: str, param_name: str, spec: dict):
    """Render sweep-mode widgets and return an Optuna search-space spec dict.

    Return format:
      integer/number → {"type": "int"|"float", "low": ..., "high": ..., "log": bool}
      enum           → {"type": "categorical", "choices": [...]}
      boolean        → {"type": "categorical", "choices": [True, False]}
    """
    enum_values = spec.get("enum")
    param_type = spec.get("type")
    default = spec.get("default")
    description = spec.get("description")
    base_key = f"{key_prefix}::sweep::{param_name}"

    # Enum → multiselect of allowed choices
    if isinstance(enum_values, list) and enum_values:
        choices = st.multiselect(
            f"{param_name} — choices",
            options=enum_values,
            default=enum_values,
            key=f"{base_key}::choices",
            help=description,
        )
        return {"type": "categorical", "choices": choices or enum_values}

    # Integer → range slider + distribution selector
    if param_type == "integer":
        schema_min = spec.get("minimum")
        schema_max = spec.get("maximum")
        default_v = _safe_int(default, 1)
        min_v = _safe_int(schema_min, 1) if schema_min is not None else 1
        max_v = (
            _safe_int(schema_max, max(default_v * 5, 100))
            if schema_max is not None
            else max(default_v * 5, 100)
        )
        if max_v <= min_v:
            max_v = min_v + 100
        default_hi = min(default_v * 2, max_v)
        if default_hi <= min_v:
            default_hi = max_v
        lo, hi = st.slider(
            f"{param_name} range",
            min_value=min_v,
            max_value=max_v,
            value=(min_v, default_hi),
            step=1,
            key=f"{base_key}::range",
            help=description,
        )
        dist = st.selectbox(
            f"{param_name} distribution",
            options=["int_uniform", "int_log_uniform"],
            key=f"{base_key}::dist",
        )
        return {"type": "int", "low": lo, "high": hi, "log": dist == "int_log_uniform"}

    # Number → two number_inputs for low/high + distribution selector
    if param_type == "number":
        schema_min = (
            spec.get("minimum") if spec.get("minimum") is not None else spec.get("exclusiveMinimum")
        )
        schema_max = (
            spec.get("maximum") if spec.get("maximum") is not None else spec.get("exclusiveMaximum")
        )
        default_v = _safe_float(default, 0.001)
        min_v = _safe_float(schema_min, 1e-6) if schema_min is not None else 1e-6
        max_v = _safe_float(schema_max, 1.0) if schema_max is not None else 1.0
        c_lo, c_hi = st.columns(2)
        with c_lo:
            lo = st.number_input(
                f"{param_name} low",
                value=float(min_v),
                format="%.6f",
                key=f"{base_key}::low",
                help=description,
            )
        with c_hi:
            hi = st.number_input(
                f"{param_name} high",
                value=float(max_v),
                format="%.6f",
                key=f"{base_key}::high",
            )
        dist = st.selectbox(
            f"{param_name} distribution",
            options=["float_uniform", "float_log_uniform"],
            key=f"{base_key}::dist",
        )
        return {"type": "float", "low": lo, "high": hi, "log": dist == "float_log_uniform"}

    # Boolean → categorical sweep over both values
    if param_type == "boolean":
        st.caption(f"{param_name}: sweep will try both True and False")
        return {"type": "categorical", "choices": [True, False]}

    # String fallback — sweep not supported; fall back to fixed widget
    st.caption(f"{param_name}: string type — sweep not supported, using fixed value")
    return _render_fixed_widget(key_prefix, param_name, spec)


def render_hyperparameters_form(schema: dict, key_prefix: str) -> dict:
    """Render a dynamic hyperparameter form driven by a JSON Schema.

    Each sweepable parameter (integer, number, enum, boolean) gets a "Sweep"
    toggle. When enabled, the standard input is replaced with a range slider
    (integer) or range number-inputs (float) plus an Optuna distribution
    selector.

    Returns a flat dict:
      - Fixed param  → {name: scalar_value}
      - Swept param  → {name: {"type": ..., "low": ..., "high": ..., "log": bool}}
                    or {name: {"type": "categorical", "choices": [...]}}
    """
    if not schema or not isinstance(schema.get("properties"), dict):
        return {}

    result: dict = {}
    for param_name, param_spec in schema["properties"].items():
        if not isinstance(param_spec, dict):
            continue

        enum_values = param_spec.get("enum")
        param_type = param_spec.get("type")
        is_sweepable = isinstance(enum_values, list) or param_type in ("integer", "number", "boolean")

        if is_sweepable:
            c_label, c_toggle = st.columns([8, 2])
            with c_label:
                st.markdown(f"**{param_name}**")
            with c_toggle:
                sweep_on = st.toggle(
                    "Sweep",
                    key=f"{key_prefix}::sweep_toggle::{param_name}",
                    value=False,
                    help=f"Enable Optuna sweep for {param_name}",
                )
            if sweep_on:
                result[param_name] = _render_sweep_widget(key_prefix, param_name, param_spec)
            else:
                result[param_name] = _render_fixed_widget(key_prefix, param_name, param_spec)
        else:
            result[param_name] = _render_fixed_widget(key_prefix, param_name, param_spec)

    return result
