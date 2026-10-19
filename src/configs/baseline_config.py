from dataclasses import dataclass
from src.utils.common import get_general_config
 
 
@dataclass
class BaselineConfig():
    n_bins: int
    smoothing: float
 
 
def get_baseline_config() -> BaselineConfig:
    general_config = get_general_config()
    baseline_config = general_config['baseline']
    return BaselineConfig(
        n_bins=baseline_config['n_bins'],
        smoothing=baseline_config['smoothing'],
    )