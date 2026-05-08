"""Tests that each model's main() sets seeds before model instantiation."""
import json
import os
from unittest.mock import MagicMock, patch, call
import pytest


JOB_SPEC = {
    "seed": 99,
    "dataset_name": "test_dataset",
    "hyperparameters": {},
    "metrics": {},
}


def _write_job_spec(tmp_path, spec=None):
    spec = spec or JOB_SPEC
    p = tmp_path / "job_spec.json"
    p.write_text(json.dumps(spec))
    return str(p)


def test_pca_sets_numpy_and_random_seeds(tmp_path):
    spec_path = _write_job_spec(tmp_path)
    with (
        patch("multiverse.models.pca.load_job_spec", return_value=JOB_SPEC),
        patch("multiverse.models.pca.load_input_mudata"),
        patch("multiverse.models.pca.anndata_concatenate"),
        patch("multiverse.models.pca.PCAModel"),
        patch("multiverse.models.pca.setup_container_logging"),
        patch("multiverse.models.pca.random") as mock_random,
        patch("multiverse.models.pca.np") as mock_np,
    ):
        from multiverse.models import pca
        pca.main()
        mock_random.seed.assert_called_once_with(99)
        mock_np.random.seed.assert_called_once_with(99)


def test_multivi_sets_scvi_seed(tmp_path):
    _write_job_spec(tmp_path)
    with (
        patch("multiverse.models.multivi.load_job_spec", return_value=JOB_SPEC),
        patch("multiverse.models.multivi.load_input_mudata"),
        patch("multiverse.models.multivi.anndata_concatenate"),
        patch("multiverse.models.multivi.MultiVIModel"),
        patch("multiverse.models.multivi.setup_container_logging"),
        patch("multiverse.models.multivi.random") as mock_random,
        patch("multiverse.models.multivi.np") as mock_np,
        patch("multiverse.models.multivi.scvi") as mock_scvi,
    ):
        from multiverse.models import multivi
        multivi.main()
        mock_random.seed.assert_called_once_with(99)
        mock_np.random.seed.assert_called_once_with(99)
        assert mock_scvi.settings.seed == 99


def test_totalvi_sets_scvi_seed(tmp_path):
    _write_job_spec(tmp_path)
    with (
        patch("multiverse.models.totalvi.load_job_spec", return_value=JOB_SPEC),
        patch("multiverse.models.totalvi.load_input_mudata"),
        patch("multiverse.models.totalvi.anndata_concatenate"),
        patch("multiverse.models.totalvi.TotalVIModel"),
        patch("multiverse.models.totalvi.setup_container_logging"),
        patch("multiverse.models.totalvi.random") as mock_random,
        patch("multiverse.models.totalvi.np") as mock_np,
        patch("multiverse.models.totalvi.scvi") as mock_scvi,
    ):
        from multiverse.models import totalvi
        totalvi.main()
        mock_random.seed.assert_called_once_with(99)
        mock_np.random.seed.assert_called_once_with(99)
        assert mock_scvi.settings.seed == 99


def test_mofa_sets_numpy_and_random_seeds(tmp_path):
    _write_job_spec(tmp_path)
    with (
        patch("multiverse.models.mofa.load_job_spec", return_value=JOB_SPEC),
        patch("multiverse.models.mofa.load_input_mudata"),
        patch("multiverse.models.mofa.MOFAModel"),
        patch("multiverse.models.mofa.setup_container_logging"),
        patch("multiverse.models.mofa.random") as mock_random,
        patch("multiverse.models.mofa.np") as mock_np,
    ):
        from multiverse.models import mofa
        mofa.main()
        mock_random.seed.assert_called_once_with(99)
        mock_np.random.seed.assert_called_once_with(99)


def test_mowgli_sets_torch_seed(tmp_path):
    _write_job_spec(tmp_path)
    with (
        patch("multiverse.models.mowgli.load_job_spec", return_value=JOB_SPEC),
        patch("multiverse.models.mowgli.load_input_mudata"),
        patch("multiverse.models.mowgli.MowgliModel"),
        patch("multiverse.models.mowgli.setup_container_logging"),
        patch("multiverse.models.mowgli.random") as mock_random,
        patch("multiverse.models.mowgli.np") as mock_np,
        patch("multiverse.models.mowgli.torch") as mock_torch,
    ):
        from multiverse.models import mowgli
        mowgli.main()
        mock_random.seed.assert_called_once_with(99)
        mock_np.random.seed.assert_called_once_with(99)
        mock_torch.manual_seed.assert_called_once_with(99)


def test_cobolt_sets_torch_seed(tmp_path):
    _write_job_spec(tmp_path)
    with (
        patch("multiverse.models.cobolt.load_job_spec", return_value=JOB_SPEC),
        patch("multiverse.models.cobolt.load_input_mudata"),
        patch("multiverse.models.cobolt.CoboltModel"),
        patch("multiverse.models.cobolt.setup_container_logging"),
        patch("multiverse.models.cobolt.random") as mock_random,
        patch("multiverse.models.cobolt.np") as mock_np,
        patch("multiverse.models.cobolt.torch") as mock_torch,
    ):
        from multiverse.models import cobolt
        cobolt.main()
        mock_random.seed.assert_called_once_with(99)
        mock_np.random.seed.assert_called_once_with(99)
        mock_torch.manual_seed.assert_called_once_with(99)


def test_seed_defaults_to_42_when_none():
    """When job_spec has no seed, models should use 42 as default."""
    spec_no_seed = {**JOB_SPEC, "seed": None}
    with (
        patch("multiverse.models.pca.load_job_spec", return_value=spec_no_seed),
        patch("multiverse.models.pca.load_input_mudata"),
        patch("multiverse.models.pca.anndata_concatenate"),
        patch("multiverse.models.pca.PCAModel"),
        patch("multiverse.models.pca.setup_container_logging"),
        patch("multiverse.models.pca.random") as mock_random,
        patch("multiverse.models.pca.np") as mock_np,
    ):
        from multiverse.models import pca
        pca.main()
        mock_random.seed.assert_called_once_with(42)
        mock_np.random.seed.assert_called_once_with(42)
