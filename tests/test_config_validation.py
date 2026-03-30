import pytest
import os
import json
from multiverse.config_schema import validate_config, SystemConfig
from pydantic import ValidationError

def test_valid_config():
    # Use a real path for testing (e.g. current directory)
    real_path = os.getcwd()
    valid_data = {
        "_run_user_params": True,
        "_run_gridsearch": False,
        "batch_key": "sample",
        "data": {
            "ds1": {
                "data_path": real_path,
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

def test_missing_batch_key():
    invalid_data = {
        "data": {"ds1": {"data_path": os.getcwd()}},
        "model": {"pca": {}}
    }
    with pytest.raises(ValidationError):
        validate_config(invalid_data)

def test_default_seed():
    data = {
        "batch_key": "batch",
        "data": {"ds1": {"data_path": os.getcwd()}},
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
