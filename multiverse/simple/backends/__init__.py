"""Execution backends for the simple-mode runner."""

from .base import ExecutionBackend, ExecutionResult
from .synthetic import SyntheticBackend

__all__ = ["ExecutionBackend", "ExecutionResult", "SyntheticBackend"]
