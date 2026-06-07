"""Integration tests for build-sif command (requires apptainer on PATH)."""

from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("apptainer") is None and shutil.which("singularity") is None,
    reason="apptainer/singularity not on PATH",
)


def test_placeholder_integration():
    """Placeholder: real integration tests require apptainer and a built image."""
    pass
