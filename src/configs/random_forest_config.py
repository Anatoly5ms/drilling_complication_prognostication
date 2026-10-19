from dataclasses import dataclass
from src.utils.common import get_general_config
 
 
@dataclass
class RandomForestConfig():
    n_estimators: int
    min_samples_leaf: int
    max_depth: int | None
    max_features: str | float
    class_weight: str | None
    random_state: int
 
 
def get_random_forest_config() -> RandomForestConfig:
    general_config = get_general_config()
    rf_config = general_config['random_forest']
    return RandomForestConfig(
        n_estimators=rf_config['n_estimators'],
        min_samples_leaf=rf_config['min_samples_leaf'],
        max_depth=rf_config['max_depth'],
        max_features=rf_config['max_features'],
        class_weight=rf_config['class_weight'],
        random_state=rf_config['random_state'],
    )