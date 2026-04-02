
import scanpy as sc
import anndata as ad
import mudata as md
import muon as mu
import os
from typing import Union
import numpy as np
from .config import load_config
#Output type = anndata, mudata

class DataLoader:
    """A data loader for reading and preprocessing single-cell datasets.

    Handles multiple file formats and converts them into standardized AnnData
    objects, ensuring necessary annotations are present.

    Attributes:
        file_path (str): The path to the data file.
        modality (str): The omics modality of the data.
        is_preprocessed (bool): Flag indicating if the input data is already preprocessed.
        annotation (str): The column name for cell type annotations.
        config_path (Union[str, dict]): Configuration object or path.
    """

    def __init__(
        self,
        file_path: str = "",
        modality: str = "",
        isProcessed: bool = True,
        annotation: str = None,
        config_path: Union[str, dict] = "./config.json",
    ):
        """Initializes the DataLoader.

        Args:
            file_path (str): The path to the data file.
            modality (str): The omics modality.
            isProcessed (bool): Whether the data is preprocessed. Defaults to True.
            annotation (str): The annotation key for cell types.
            config_path (Union[str, dict]): Configuration path or dictionary.
        """
        self.config_path = config_path
        self.file_path = file_path
        self.modality = modality
        self.is_preprocessed = isProcessed
        self.annotation = annotation
        self.data = None

    def read_anndata(self) -> ad.AnnData:
        """Reads the specified file into an AnnData object.

        Supported formats include: `.csv`, `.tsv`, `.h5ad`, `.txt`, `.mtx`, `.h5mu`, and `.h5`.

        Returns:
            ad.AnnData: The loaded (unprocessed) AnnData object.

        Raises:
            ValueError: If the file format is unsupported or if modality and file_path are missing.
        """
        adata = None
        # Modality and file_path should be provided to load anndata object
        if self.modality != "" and self.file_path != "": 
            if ".csv" in self.file_path:
                adata = sc.read_csv(self.file_path)
            elif ".tsv" in self.file_path:
                adata = sc.read(self.file_path, delimiter='\t').T
            elif ".h5ad" in self.file_path:
                adata = sc.read_h5ad(self.file_path)  
            elif ".txt" in self.file_path:
                adata = sc.read_text(self.file_path)
            elif ".mtx" in self.file_path:
                if self.modality in ["rna", "atac"]:
                    path = os.path.dirname(self.file_path)
                    mudata = mu.read_10x_mtx(path, extended=True) 
                    adata = mudata[self.modality]
                else:
                    adata = sc.read_mtx(self.file_path)
            elif ".h5mu" in self.file_path:
                mudata = mu.read_h5mu(self.file_path)
                adata = mudata[self.modality]
                # Propagate specific metadata keys for MuData files to maintain batch info.
                adata.obs["batch"] = mudata.obs["batch"]
                adata.obs["mod_id"] = mudata.obs["mod_id"]
            elif ".h5" in self.file_path:
                mudata =mu.read_10x_h5(self.file_path)
                adata = mudata[self.modality]

            if adata:  # Check if adata is not None
                adata.var_names_make_unique()

                # Standardize cell type annotations to 'cell_type' key for downstream plotting.
                if self.annotation is None:
                    self.annotation = "cell_type"
                    num_obs = adata.n_obs
                    adata.obs[self.annotation] = np.zeros(num_obs, dtype=int)
                elif self.annotation != "cell_type":
                    adata.obs["cell_type"] = adata.obs[self.annotation]
                    self.annotation = "cell_type"

                self.data = adata
                return self.data
            else:
                raise ValueError("Could not read the file. Please check the file path and format.")
        else:
            raise ValueError("Modality and file_path must be provided for anndata loading.")

    def read_mudata(self) -> md.MuData:
        """Reads a file into a MuData object.

        Returns:
            md.MuData: The loaded MuData object.

        Raises:
            ValueError: If the file path is empty or format is unsupported.
        """
        if self.file_path != "":
            if ".h5mu" in self.file_path:
                mudata = mu.read(self.file_path)
            elif ".h5" in self.file_path:
                mudata =mu.read_10x_h5(self.file_path)
            elif ".mtx" in self.file_path:
                path = os.path.dirname(self.file_path)
                mudata = mu.read_10x_mtx(path, extended=True) 
            else:
                raise ValueError("Could not read the file. Only support file format .h5mu, .h5, .mtx.")
        else:
            raise ValueError("file_path must be provided to read mudata files")
        
        self.data = mudata
        return self.data

    def preprocessing(self) -> ad.AnnData:
        """Loads and preprocesses the AnnData object.

        If the data is not marked as preprocessed, the appropriate preprocessing
        routine (RNA, ATAC, or ADT) is triggered.

        Returns:
            ad.AnnData: The preprocessed AnnData object.

        Raises:
            ValueError: If the file path, modality is missing or preprocessing is not applicable.
        """
         # Modality and file_path should be provided for the read_anndata() function to work
        if self.file_path != "":
            if self.modality != "" :
                self.read_anndata()
                self.data.var_names_make_unique()
                self.data.layers["counts"] = self.data.X.copy()
                if not self.is_preprocessed:
                    pre = Preprocessing(anndata=self.data, config_path=self.config_path)
                    # RNA preprocessing
                    if self.modality=="rna":
                        self.data = pre.rna_preprocessing()
                    # ATAC preprocessing
                    elif self.modality=="atac":
                        self.data = pre.atac_preprocessing()
                    # ADT preprocessing
                    elif self.modality=="adt":
                        self.data = pre.adt_preprocessing()
                    # Not applicable
                    else:
                        raise ValueError("Preprocessing for this modality is not applicable!")
            else:
                raise ValueError("Modality must be provided to read anndata")
        else:
            raise ValueError("File_path must be provided to be read")
        return self.data


class Preprocessing:
    """Technical preprocessing routines for different single-cell modalities.

    Attributes:
        data (ad.AnnData): The AnnData object to be preprocessed.
        config (dict): The modality-specific preprocessing configuration.
    """

    def __init__(
        self, anndata: ad.AnnData, config_path: Union[str, dict] = "./config.json"
    ):
        """Initializes the Preprocessing object.

        Args:
            anndata (ad.AnnData): The dataset to process.
            config_path (Union[str, dict]): Configuration path or dictionary.
        """
        self.data = anndata
        self.config = load_config(config_path).get("preprocess_params")

    def rna_preprocessing(self) -> ad.AnnData:
        """Performs quality control, normalization, and feature selection for RNA data.

        Returns:
            ad.AnnData: The preprocessed RNA dataset.
        """

        rna_dict = self.config.get("rna_filtering")

        # Quality control - based on scanpy calculateQCmetrics - McCarthy et al., 2017
        self.data.var["mt"] = self.data.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(self.data, qc_vars=["mt"], inplace=rna_dict.get("qc_metric_inplace"), log1p=rna_dict.get("qc_metric_log1p"))
        
        # Filtering -> threshold metrics depend on the specific dataset and experimental conditions.
        mu.pp.filter_obs(self.data, 'n_genes_by_counts', lambda x: (x >= rna_dict.get("min_genes_by_counts")) & (x < rna_dict.get("max_genes_by_counts")))
        mu.pp.filter_obs(self.data, "total_counts", lambda x: x < rna_dict.get("max_total_counts_per_cell"))
        mu.pp.filter_obs(self.data, "pct_counts_mt", lambda x: x < rna_dict.get("max_pct_counts_mt"))

        # Filter genes by keeping only those that are expressed in at least 10 cells.
        mu.pp.filter_var(self.data, "n_cells_by_counts", lambda x: x >= rna_dict.get("min_cells_by_counts"))        
        
        # Normalisation
        sc.pp.normalize_total(self.data, target_sum=rna_dict.get("normalization_target_sum"))
        sc.pp.log1p(self.data)

        # Feature selection
        sc.pp.highly_variable_genes(
            self.data,
            n_top_genes=rna_dict.get("n_top_genes"),
            subset=True,
            flavor="seurat",
        )

        return self.data

    def atac_preprocessing(self) -> ad.AnnData:
        """Performs quality control, normalization, and feature selection for ATAC data.

        Returns:
            ad.AnnData: The preprocessed ATAC dataset.
        """

        atac_dict = self.config.get("atac_filtering")

        # Quality control
        sc.pp.calculate_qc_metrics(self.data, percent_top=None, inplace=atac_dict.get("qc_metric_inplace"), log1p=atac_dict.get("qc_metric_log1p"))

        # Filter cells based on QC metrics.
        mu.pp.filter_obs(self.data, "n_genes_by_counts", lambda x: (x >= atac_dict.get("min_peaks_by_counts")) & (x <= atac_dict.get("max_peaks_by_counts")))
        mu.pp.filter_obs(self.data, "total_counts", lambda x: (x >= atac_dict.get("min_total_counts_per_cell")) & (x <= atac_dict.get("max_total_counts_per_cell")))
        
        # Filter peaks based on number of cells where they are present.
        mu.pp.filter_var(self.data, "n_cells_by_counts", lambda x: x < atac_dict.get("max_cells_by_counts"))
        mu.pp.filter_var(self.data, "total_counts", lambda x: x < atac_dict.get("max_total_counts_by_gene"))

        # Perform per-cell normalization.
        sc.pp.normalize_total(self.data, target_sum=atac_dict.get("normalization_target_sum"))
        sc.pp.log1p(self.data)

        # Feature selection
        sc.pp.highly_variable_genes(
            self.data,
            n_top_genes=atac_dict.get("n_top_peaks"),
            subset=True,
            flavor="seurat",
        )

        return self.data

    def adt_preprocessing(self) -> ad.AnnData:
        """Performs normalization for ADT data.

        Returns:
            ad.AnnData: The preprocessed ADT dataset.
        """

        adt_dict = self.config.get("adt_filtering")

        # Remove the "total" feature.
        self.data = self.data[:, 1:]
        # Make index of proteins compatible with 10X multiome.
        self.data.obs.index += "-1"

        # Perform per-cell normalization.
        if adt_dict.get("per_cell_normalization"):
            mu.prot.pp.clr(self.data)
        self.data.var["highly_variable"] = True
        self.data.var["feature_types"] = "Protein Expression"
        self.data.var["genome"] = "GRCh38"

        return self.data

