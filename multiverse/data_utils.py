import anndata as ad
import mudata as md
import muon as mu
import numpy as np
import os
from typing import List, Union
from .logging_utils import get_logger
from .dataloader import DataLoader
from .config import load_config


logger = get_logger(__name__)


def fuse_mudata(list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None) -> md.MuData:
    """Fuses a list of AnnData objects into a single MuData object.

    Uses `muon.pp.intersect_obs` to ensure that the observation indices are consistent across
    all modalities.

    Args:
        list_anndata (List[ad.AnnData]): A list of AnnData objects to be fused.
        list_modality (List[str]): A list of modality names corresponding to each AnnData.

    Returns:
        mudata.MuData: The fused MuData object containing all modalities.

    Raises:
        ValueError: If the lengths of `list_modality` and `list_anndata` are not equal.
    """
    if len(list_modality) != len(list_anndata):
        raise ValueError("Length of list_modality and list_anndata must be equal!")
    else:
        data_dict = {}
        for i, mod in enumerate(list_modality):
            data_dict[mod] = list_anndata[i]
            try:
                list_anndata[i].X = np.array(list_anndata[i].X.todense())
            except AttributeError:
                pass

    data = mu.MuData(data_dict)
    mu.pp.intersect_obs(data)   # Make sure number of cells are the same for all modalities

    # Ensures consistency in cell type annotations by using 'rna' modality's labels
    # if available, otherwise creates a default column.
    if "cell_type" in data["rna"].obs.columns:
        data.obs["cell_type"] = data["rna"].obs["cell_type"]
    else:
        logger.warning("No 'cell_type' annotation found in rna modality. Creating a default 'cell_type' column with zeros.")
        num_obs = data.n_obs
        data.obs["cell_type"] = np.zeros(num_obs, dtype=int)

    return data

def anndata_concatenate(list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None) -> ad.AnnData:
    """Concatenates multiple AnnData objects along the variable axis.

    First fuses the data into a MuData object to ensure alignment of observations,
    then concatenates the individual modalities into a single AnnData object.

    Args:
        list_anndata (List[ad.AnnData]): A list of AnnData objects to be concatenated.
        list_modality (List[str]): A list of modality names.

    Returns:
        anndata.AnnData: The concatenated AnnData object.
    """
    mudata = fuse_mudata(list_anndata=list_anndata, list_modality=list_modality)
    list_ann = []
    for mod in list_modality:
        list_ann.append(mudata[mod])

    anndata = ad.concat(list_ann, axis="var", label="cell_type", merge="unique", uns_merge="unique")

    # Propagates cell type annotations from the fused MuData to the concatenated object.
    num_obs = anndata.n_obs
    if "cell_type" in mudata.obs.columns:
        anndata.obs["cell_type"] = mudata.obs["cell_type"]
    else:
        anndata.obs["cell_type"] = np.zeros(num_obs, dtype=int)
    
    # Initialize modality mapping for compatibility with models like MultiVI.
    anndata.obs["modality"] = np.zeros(num_obs, dtype=int)
    return anndata


def load_datasets(config_path_or_dict: Union[str, dict]):
    """Loads and preprocesses all datasets specified in the system configuration.

    Args:
        config_path_or_dict: Path to the JSON configuration file or the configuration dict.

    Returns:
        dict: A dictionary where keys are dataset names and values are dictionaries
            containing modality names and the loaded AnnData objects.

    Raises:
        ValueError: If a dataset has no modalities defined.
    """
    config_dict = load_config(config_path_or_dict)
    datasets = {}

    data_config = config_dict.get("data", {})
    dataset_names = [
        key
        for key, value in data_config.items()
        if isinstance(value, dict) and "data_path" in value
    ]

    for dataset_name in dataset_names:
        dataset_info = data_config[dataset_name]
        modality_list = [
            key
            for key, value in dataset_info.items()
            if isinstance(value, dict) and "file_name" in value
        ]
        dataset_path = dataset_info["data_path"]
        list_anndata = []
        # Check if data is loaded correctly
        if modality_list is not None:
            for modality in modality_list:
                modality_info = dataset_info[modality]
                file_path = os.path.join(dataset_path, modality_info["file_name"])
                is_preprocessed = modality_info["is_preprocessed"]
                annotation = modality_info["annotation"]
                ann_loader = DataLoader(
                    file_path=file_path,
                    modality=modality,
                    isProcessed=is_preprocessed,
                    annotation=annotation,
                    config_path=config_path_or_dict,
                )
                ann = ann_loader.preprocessing()
                list_anndata.append(ann)
            datasets[dataset_name] = {"modalities": modality_list, "data": list_anndata}
        else:
            raise ValueError("Modality is None. Trainer not applicable for this case.")
    return datasets


def dataset_select(datasets_dict: dict, data_type: str = ""):
    """Converts the internal dataset dictionary into the format required by models.

    Args:
        datasets_dict (dict): The dictionary returned by `load_datasets`.
        data_type (str): The desired output format, either "concatenate" (for AnnData)
            or "mudata" (for MuData).

    Returns:
        dict: A dictionary mapping dataset names to the processed data objects.

    Raises:
        ValueError: If an unsupported `data_type` is provided.
    """
    datasets = datasets_dict

    if data_type == "concatenate":
        # PCA and MultiVI require a single AnnData with concatenated modalities.
        concatenate = {}
        for dataset_name, dataset_data in datasets.items():
            logger.info(f"Concatenating dataset: {dataset_name}")
            modalities = dataset_data["modalities"]
            list_anndata = dataset_data["data"]
            data_concat = anndata_concatenate(
                list_anndata=list_anndata, list_modality=modalities
            )
            concatenate[dataset_name] = data_concat
        data = concatenate
    elif data_type == "mudata":
        # MOFA+ and Mowgli use MuData to represent multiple modalities separately.
        mudata_input = {}
        for dataset_name, dataset_data in datasets.items():
            logger.info(f"Fusing dataset as MuData object: {dataset_name}")
            modalities = dataset_data["modalities"]
            list_anndata = dataset_data["data"]
            data_fuse = fuse_mudata(list_anndata=list_anndata, list_modality=modalities)
            mudata_input[dataset_name] = data_fuse
        data = mudata_input
    else:
        raise ValueError("Only accept datatype of concatenate or mudata.")
    return data
