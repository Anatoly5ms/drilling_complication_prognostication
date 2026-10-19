from dataclasses import dataclass
from src.utils.common import get_general_config


@dataclass
class ModelSelectorConfig():
    reference_model: str
    min_ensemble_size: int


def get_model_selector_config() -> ModelSelectorConfig:
    general_config = get_general_config()
    selection_config = general_config['model_selection']
    return ModelSelectorConfig(
        reference_model=selection_config['reference_model'],
        min_ensemble_size=selection_config['min_ensemble_size'],
    )