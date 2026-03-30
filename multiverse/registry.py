import json
import os
from typing import List, Dict, Set
from pydantic import BaseModel, Field
from .logging_utils import get_logger

logger = get_logger(__name__)

class ModelEntry(BaseModel):
    name: str
    docker_image: str
    supported_omics: List[str]

class ModelRegistry(BaseModel):
    models: List[ModelEntry]

def load_registry(registry_path: str = "model_registry.json") -> Dict[str, ModelEntry]:
    """
    Load the model registry from a JSON file.
    Returns a dictionary mapping model names to ModelEntry objects.
    """
    if not os.path.exists(registry_path):
        # Try relative to the project root if not found
        root_registry = os.path.join(os.path.dirname(os.path.dirname(__file__)), registry_path)
        if os.path.exists(root_registry):
            registry_path = root_registry
        else:
            raise FileNotFoundError(f"Model registry file not found at {registry_path}")

    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
        registry = ModelRegistry(**data)
        logger.info(f"Loaded {len(registry.models)} models from registry.")
        return {m.name: m for m in registry.models}
    except Exception as e:
        logger.error(f"Failed to load model registry: {e}")
        raise

def get_eligible_models(
    user_requested_models: List[str],
    available_omics: List[str],
    registry: Dict[str, ModelEntry]
) -> List[str]:
    """
    Match the extracted dataset omics to the available models in the registry.
    A model is eligible if all its supported_omics are present in available_omics.
    (Or if it supports a subset, depending on requirements. Task T3.2 says
    'check if the model's required omics are a subset or match the dataset's available omics')
    """
    eligible_models = []
    available_set = set(available_omics)

    for model_name in user_requested_models:
        if model_name not in registry:
            logger.warning(f"Model '{model_name}' requested but not found in registry. Skipping.")
            continue

        model_entry = registry[model_name]
        required_set = set(model_entry.supported_omics)

        # Check if required omics are available in the dataset
        if required_set.issubset(available_set):
            eligible_models.append(model_name)
            logger.info(f"Model '{model_name}' is eligible.")
        else:
            missing = required_set - available_set
            logger.warning(f"Model '{model_name}' is ineligible. Missing omics: {missing}")

    return eligible_models
