import anndata as ad
import mudata as md
import muon as mu
import numpy as np
import os
from typing import List
from .logging_utils import get_logger
from .dataloader import DataLoader
from .config import load_config


logger = get_logger(__name__)


def fuse_mudata(list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None) -> md.MuData:
    """
    Fusing paired anndata as MuData
    intersect_obs will be used if number of obs not equivalent
    Args:
        list_modality: A list of strings representing the modalities (e.g., ["rna", "atac", "adt"]).
        list_anndata: A list of AnnData objects corresponding to the modalities (e.g., [adata_rna, adata_atac, adata_adt]).
    Returns:s
        A mudata.MuData object    
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

    # Hard-code "cell_type" to avoid conflict
    if "cell_type" in data["rna"].obs.columns:
        data.obs["cell_type"] = data["rna"].obs["cell_type"]
    else:
        # If there is no 'cell_type' annotation in rna modality
        logger.warning("No 'cell_type' annotation found in rna modality. Creating a default 'cell_type' column with zeros.")
        num_obs = data.n_obs
        data.obs["cell_type"] = np.zeros(num_obs, dtype=int)

    return data

def anndata_concatenate(list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None) -> ad.AnnData:
    """
    Args:
        list_modality: A list of strings representing the modalities (e.g., ["rna", "atac", "adt"]).
        list_anndata: A list of AnnData objects corresponding to the modalities (e.g., [adata_rna, adata_atac, adata_adt]).
    Returns:
        A AnnData object
    """
    mudata = fuse_mudata(list_anndata=list_anndata, list_modality=list_modality)
    list_ann = []
    for mod in list_modality:
        list_ann.append(mudata[mod])

    anndata = ad.concat(list_ann, axis="var", label="cell_type", merge="unique", uns_merge="unique")

    # Hard-code "cell_type" to avoid conflict (Should already be available when fuse_mudata is called above)
    num_obs = anndata.n_obs
    if "cell_type" in mudata.obs.columns:
        anndata.obs["cell_type"] = mudata.obs["cell_type"]
    else:
        # No annotation -> setting annotation as 'cell_type' = 0 to avoid conflicts.
        anndata.obs["cell_type"] = np.zeros(num_obs, dtype=int)
    
    anndata.obs["modality"] = np.zeros(num_obs, dtype=int) # Adding this to prevent error in multiVI model
    return anndata


def load_datasets(config_path):
    """
    Load all datasets specified in the configuration.
    Returns a dictionary where keys are dataset names, and values are the data objects.
    """
    config_dict = load_config(config_path)
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
        ]  # modality (i.e.'rna') must be a dictionary
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
                    config_path=config_path,
                )
                ann = ann_loader.preprocessing()
                list_anndata.append(ann)
            datasets[dataset_name] = {"modalities": modality_list, "data": list_anndata}
        else:
            raise ValueError("Modality is None. Trainer not applicable for this case.")
    return datasets


def dataset_select(datasets_dict, data_type: str = ""):
    """
    Concatenate list of AnnDatas or Fuse list of AnnDatas into one MuData
    """
    datasets = datasets_dict

    if data_type == "concatenate":  # Process input object for PCA and MultiVI
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
    elif data_type == "mudata":  # Process input object for MOFA+ and Mowgli
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
