import scanpy as sc
import mudata as md
import muon as mu
import os
from typing import List, Union, Optional
from .logging_utils import get_logger

logger = get_logger(__name__)

def load_dataset(file_path: str) -> Union[sc.AnnData, md.MuData]:
    """Loads a single-cell dataset from a file.

    Supported formats include `.h5ad` for AnnData and `.h5mu` for MuData.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        Union[scanpy.AnnData, mudata.MuData]: The loaded dataset object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    if file_path.endswith(".h5ad"):
        logger.info(f"Loading AnnData from {file_path}")
        return sc.read_h5ad(file_path)
    elif file_path.endswith(".h5mu"):
        logger.info(f"Loading MuData from {file_path}")
        return mu.read_h5mu(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .h5ad or .h5mu.")

def validate_dataset_structure(
    data: Union[sc.AnnData, md.MuData],
    batch_key: str,
    cell_type_key: Optional[str] = None
) -> List[str]:
    """Verifies internal structural requirements of the dataset.

    Ensures that the specified `batch_key` and `cell_type_key` (if provided)
    exist in the dataset observations (`.obs`).

    Args:
        data (Union[sc.AnnData, md.MuData]): The dataset object to validate.
        batch_key (str): The observation key identifying experimental batches.
        cell_type_key (Optional[str]): The observation key identifying cell types.
            Defaults to None.

    Returns:
        List[str]: A list of available omics (modalities) in the dataset.

    Raises:
        ValueError: If required keys are missing from the dataset observations.
        TypeError: If the input data is not an AnnData or MuData object.
    """
    # Check if keys exist in observations
    if batch_key not in data.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in dataset observations.")

    if cell_type_key and cell_type_key not in data.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in dataset observations.")

    # Extract available omics
    if isinstance(data, md.MuData):
        omics = list(data.mod.keys())
    elif isinstance(data, sc.AnnData):
        # Default to rna if it's an AnnData object
        omics = ["rna"]
    else:
        raise TypeError("Dataset must be an AnnData or MuData object.")

    logger.info(f"Dataset validated. Available omics: {omics}")
    return omics
