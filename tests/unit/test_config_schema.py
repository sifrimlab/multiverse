import pytest
import os
from multiverse.config_schema import validate_config
from pydantic import ValidationError

def test_valid_config(tmp_path):
    # Use tmp_path for data_path to ensure it exists
    valid_data = {
        "_run_user_params": True,
        "_run_gridsearch": False,
        "batch_key": "sample",
        "data": {
            "ds1": {
                "data_path": str(tmp_path),
                "rna": {"file_name": "rna.h5ad"}
            }
        },
        "model": {
            "pca": {"n_components": 20}
        }
    }
    config = validate_config(valid_data)
    assert config.batch_key == "sample"
    assert config.random_seed == 42 # Default

def test_missing_batch_key(tmp_path):
    invalid_data = {
        "data": {"ds1": {"data_path": str(tmp_path)}},
        "model": {"pca": {}}
    }
    with pytest.raises(ValidationError):
        validate_config(invalid_data)

def test_default_seed(tmp_path):
    data = {
        "batch_key": "batch",
        "data": {"ds1": {"data_path": str(tmp_path)}},
        "model": {"pca": {}}
    }
    config = validate_config(data)
    assert config.random_seed == 42

def test_invalid_path():
    invalid_data = {
        "batch_key": "batch",
        "data": {
            "real_ds": {
                "data_path": "/non/existent/path"
            }
        },
        "model": {"pca": {}}
    }
    with pytest.raises(ValidationError):
        validate_config(invalid_data)
