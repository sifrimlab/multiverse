"""
Integration tests for the multiverse pipeline.

This test suite validates that the multiverse workflow can run end-to-end
with minimal test data.
"""

import os
import json
import pytest
from tools.legacy_local_runner import main_workflow


@pytest.fixture
def mock_config_path():
    """Provide path to the test configuration file."""
    return os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'mock_config.json')


def test_integration_minimal_run(mock_config_path, test_data_dir, outputs_dir):
    """
    Test that the pipeline can run with minimal dummy data.
    
    This test:
    1. Loads a minimal "dummy" config
    2. Runs runner.py workflow for 1 model (PCA) on tiny dataset
    3. Asserts that outputs/evaluation_metrics.json is created
    """
    # Update config to point to test data
    with open(mock_config_path, 'r') as f:
        config = json.load(f)
    
    # Update data path to test directory
    config['data']['dataset_test']['data_path'] = test_data_dir
    config['output_dir'] = outputs_dir
    
    # Create temporary config file with updated paths
    temp_config_path = os.path.join(test_data_dir, 'temp_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run the workflow
    main_workflow(temp_config_path)
    
    # Verify outputs exist
    assert os.path.exists(outputs_dir), "Outputs directory was not created"
    
    # Check that the dataset output directory exists
    dataset_output_dir = os.path.join(outputs_dir, 'dataset_test')
    assert os.path.exists(dataset_output_dir), f"Dataset output directory not created: {dataset_output_dir}"
    
    # Check that PCA model outputs exist
    pca_output_dir = os.path.join(dataset_output_dir, 'pca')
    assert os.path.exists(pca_output_dir), f"PCA output directory not created: {pca_output_dir}"
    
    # Check for embeddings file
    embeddings_path = os.path.join(pca_output_dir, 'embeddings.h5')
    assert os.path.exists(embeddings_path), f"Embeddings file not created: {embeddings_path}"
    
    # Check for UMAP visualization
    umap_path = os.path.join(pca_output_dir, 'umap.png')
    assert os.path.exists(umap_path), f"UMAP file not created: {umap_path}"
    
    # Check for evaluation metrics
    metrics_path = os.path.join(dataset_output_dir, 'evaluation_metrics.json')
    assert os.path.exists(metrics_path), f"Evaluation metrics file not created: {metrics_path}"
    
    # Verify that metrics file contains valid JSON
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        assert isinstance(metrics, dict), "Metrics should be a dictionary"


def test_config_loading(mock_config_path):
    """Test that the test config can be loaded successfully."""
    with open(mock_config_path, 'r') as f:
        config = json.load(f)
    
    assert config is not None
    assert '_run_user_params' in config
    assert config['_run_user_params'] is True
    assert 'model' in config
    assert 'pca' in config['model']
