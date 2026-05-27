"""Move-1 exit-gate test: the control-plane test subset is importable
without the ML-legacy stack.

Strategy v2 "Current Gaps and Next Moves" §1: control-plane tests
(artifact, journal, simple, promotion, mvd, index, doctor, gc, broker,
registration, client) must run on a minimal dev environment without
``scanpy``.

This test imports each control-plane test module in a subprocess where
``scanpy``/``mudata``/``anndata`` are forced absent via a finder hook, and
asserts the import succeeds. Test *fixtures* may still call into ML
packages — those fixtures use ``pytest.importorskip`` so collection
succeeds either way.

(``test_artifact_validation`` and ``test_simple_mode`` and a handful of
other tests use ``h5py``/``numpy`` directly for synthesizing embeddings;
``h5py`` and ``numpy`` are NOT ML-legacy and remain required.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


CONTROL_PLANE_TEST_MODULES = (
    "tests.unit.test_artifact_contract",
    "tests.unit.test_artifact_validation",
    "tests.unit.test_simple_mode",
    "tests.unit.test_journal",
    "tests.unit.test_promotion_saga",
    "tests.unit.test_docker_supervisor",
    "tests.unit.test_mvd_kernel",
    "tests.unit.test_index_rebuild",
    "tests.unit.test_client_cutover",
    "tests.unit.test_projection_sync",
    "tests.unit.test_doctor",
    "tests.unit.test_gc",
    "tests.unit.test_broker",
    "tests.unit.test_registration_hardening",
)


_BLOCK_ML_PKGS = ("scanpy", "mudata", "anndata", "scvi", "muon", "torch")


def _block_script(target_module: str) -> str:
    """Build a script that blocks ML-stack imports then imports a test
    module's `multiverse.*` dependencies.

    Importing the test module itself would trigger pytest collection
    machinery + ``conftest.py``; instead we import every ``multiverse.``
    package the strategy lists as control-plane and a few cross-cutting
    helpers (h5py, numpy) the tests actually use at module level.
    """
    block_list = ",".join(repr(m) for m in _BLOCK_ML_PKGS)
    return (
        "import sys\n"
        "_blocked = {" + block_list + "}\n"
        "class _BlockFinder:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.', 1)[0]\n"
        "        if top in _blocked:\n"
        "            raise ImportError(f'ML dep {top!r} is intentionally blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockFinder())\n"
        "import importlib\n"
        f"importlib.import_module({target_module!r})\n"
        "print('ok')\n"
    )


_CONTROL_PLANE_PACKAGES = (
    "multiverse.artifact",
    "multiverse.journal",
    "multiverse.simple",
    "multiverse.promotion",
    "multiverse.docker_supervisor",
    "multiverse.mvd",
    "multiverse.index",
    "multiverse.client",
    "multiverse.projection",
    "multiverse.doctor",
    "multiverse.gc",
    "multiverse.broker",
    "multiverse.registration",
)


@pytest.mark.control_plane
@pytest.mark.parametrize("package", _CONTROL_PLANE_PACKAGES)
def test_control_plane_package_imports_without_ml_stack(package: str) -> None:
    """Each control-plane package must import even if every ML-legacy
    package is unavailable."""
    result = subprocess.run(
        [sys.executable, "-c", _block_script(package)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{package} failed to import with ML blocked.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )


@pytest.mark.control_plane
def test_conftest_does_not_import_ml_stack_at_module_load() -> None:
    """Importing tests.conftest must not load scanpy / mudata / anndata.

    Importing conftest also imports its own globals — if a `scanpy` line
    reappears at the top, this subprocess invocation will fail.
    """
    script = (
        "import sys\n"
        # The conftest lives at tests/conftest.py; import via importlib.
        "import importlib.util\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('tests_conftest', "
        f"{str(Path(__file__).resolve().parents[1] / 'conftest.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "leaked = [m for m in ('scanpy', 'mudata', 'anndata') if m in sys.modules]\n"
        "if leaked:\n"
        "    print(','.join(leaked))\n"
        "    raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"conftest leaked ML imports: {result.stdout.strip()!r}\n"
        f"stderr: {result.stderr}"
    )
