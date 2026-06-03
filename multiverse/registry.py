import json
from typing import Dict, List

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


def generate_compatibility_matrix(
    datasets: List[Dict], models: List[Dict]
) -> pd.DataFrame:
    """Generates a compatibility matrix between datasets and models based on omics.

    Args:
        datasets (List[Dict]): List of dataset metadata dictionaries from the registry.
        models (List[Dict]): List of model metadata dictionaries from the registry.

    Returns:
        pd.DataFrame: A DataFrame with Datasets as rows and Models as columns.
    """
    matrix_data = []
    dataset_names = [d["name"] for d in datasets]
    model_names = [m["name"] for m in models]

    for d in datasets:
        row = []
        # Handle both list and JSON string formats from SQLite
        available_omics = d.get("omics_available", [])
        if isinstance(available_omics, str):
            available_omics = json.loads(available_omics)
        available_set = set(available_omics)

        for m in models:
            supported_omics = m.get("supported_omics", [])
            if isinstance(supported_omics, str):
                supported_omics = json.loads(supported_omics)
            supported_set = set(supported_omics)

            if "any" in supported_set:
                status = "Compatible"
            elif supported_set.issubset(available_set):
                if supported_set == available_set:
                    status = "Compatible"
                else:
                    status = "Partial"
            else:
                status = "Incompatible"
            row.append(status)
        matrix_data.append(row)

    return pd.DataFrame(matrix_data, index=dataset_names, columns=model_names)
