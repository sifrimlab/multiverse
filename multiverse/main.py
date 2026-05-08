"""Compatibility shim for the relocated legacy local runner."""

from tools.legacy_local_runner import main_workflow, run_models_with_user_params, set_seed

__all__ = ["main_workflow", "run_models_with_user_params", "set_seed"]

