import os
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ModalityConfig(BaseModel):
    file_name: str
    is_preprocessed: bool = False
    annotation: Optional[str] = None


class DatasetConfig(BaseModel):
    data_path: str
    rna: Optional[ModalityConfig] = None
    atac: Optional[ModalityConfig] = None
    adt: Optional[ModalityConfig] = None

    @field_validator("data_path")
    @classmethod
    def validate_path_exists(cls, v):
        if not os.path.exists(v):
             raise ValueError(f"Data path {v} does not exist.")
        return v


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    device: str = "cpu"
    umap_random_state: int = 42
    umap_color_type: Optional[str] = None
    grid_search_params: Dict[str, Any] = Field(default_factory=dict)


class PreprocessParams(BaseModel):
    rna_filtering: Dict[str, Any] = Field(default_factory=dict)
    atac_filtering: Dict[str, Any] = Field(default_factory=dict)
    adt_filtering: Dict[str, Any] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_user_params: bool = Field(True, alias="_run_user_params")
    run_gridsearch: bool = Field(False, alias="_run_gridsearch")
    data: Dict[str, DatasetConfig]
    model: Dict[str, ModelParams]
    preprocess_params: Optional[PreprocessParams] = None
    training: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "./outputs/"
    random_seed: int = 42
    batch_key: str  # Mandatory as per task description, no default

    @field_validator("data")
    @classmethod
    def validate_datasets(cls, v):
        for name, ds in v.items():
            if not os.path.exists(ds.data_path):
                raise ValueError(f"Data path {ds.data_path} for dataset {name} does not exist.")
        return v

def validate_config(config_data: Dict[str, Any]) -> SystemConfig:
    """Validates the configuration dictionary against the Pydantic schema.

    Args:
        config_data (Dict[str, Any]): The raw configuration dictionary.

    Returns:
        SystemConfig: A validated SystemConfig object.

    Raises:
        pydantic.ValidationError: If the configuration does not match the schema.
    """
    return SystemConfig(**config_data)
