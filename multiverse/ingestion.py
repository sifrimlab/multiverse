import scanpy as sc
import mudata as md
import muon as mu
import os
from typing import List, Union, Optional
from .logging_utils import get_logger

logger = get_logger(__name__)

def load_dataset(file_path: str) -> Union[sc.AnnData, md.MuData]:
    """
    Load data from h5ad or h5mu format.
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
    """
    Verify internal structural requirements.
    Ensures batch_key and cell_type_key (if provided) exist in observations.
    Returns list of available omics.
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
