"""
Integration tests for the multiverse pipeline.

This test suite validates that the multiverse workflow can run end-to-end
with minimal test data.
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
import numpy as np
import anndata as ad

# Add the parent directory to the path so we can import multiverse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiverse.main import main_workflow


@pytest.fixture
def test_config_path():
    """Provide path to the test configuration file."""
    return os.path.join(os.path.dirname(__file__), 'test_config_dummy.json')


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with minimal test data."""
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix='mvexp_test_')
    data_dir = os.path.join(temp_dir, 'test_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create minimal dummy RNA data (10 cells x 100 genes)
    n_obs = 10
    n_vars = 100
    
    # Create random expression matrix
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    
    # Add required annotations
    adata.obs['cell_type'] = np.random.choice(['TypeA', 'TypeB'], size=n_obs)
    
    # Save to h5ad file
    adata.write_h5ad(os.path.join(data_dir, 'test_rna.h5ad'))
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def outputs_dir():
    """Setup and cleanup outputs directory."""
    output_dir = './outputs/'
    
    # Clean up before test
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    yield output_dir
    
    # Clean up after test
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_integration_minimal_run(test_config_path, test_data_dir, outputs_dir):
    """
    Test that the pipeline can run with minimal dummy data.
    
    This test:
    1. Loads a minimal "dummy" config
    2. Runs runner.py workflow for 1 model (PCA) on tiny dataset
    3. Asserts that outputs/evaluation_metrics.json is created
    """
    # Update config to point to test data
    with open(test_config_path, 'r') as f:
        config = json.load(f)
    
    # Update data path to test directory
    config['data']['dataset_test']['data_path'] = test_data_dir
    
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
    
    # Check for evaluation metrics (not results.json as per the problem statement,
    # but evaluation_metrics.json as per actual code)
    metrics_path = os.path.join(dataset_output_dir, 'evaluation_metrics.json')
    assert os.path.exists(metrics_path), f"Evaluation metrics file not created: {metrics_path}"
    
    # Verify that metrics file contains valid JSON
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        assert isinstance(metrics, dict), "Metrics should be a dictionary"


def test_config_loading(test_config_path):
    """Test that the test config can be loaded successfully."""
    with open(test_config_path, 'r') as f:
        config = json.load(f)
    
    assert config is not None
    assert '_run_user_params' in config
    assert config['_run_user_params'] is True
    assert 'model' in config
    assert 'pca' in config['model']
