from dataclasses import dataclass
from src.utils.common import get_general_config


@dataclass
class CatBoostConfig():
    iterations: int
    learning_rate: float
    depth: int
    l2_leaf_reg: float
    auto_class_weights: str | None
    random_seed: int


def get_catboost_config() -> CatBoostConfig:
    general_config = get_general_config()
    catboost_config = general_config['catboost']
    return CatBoostConfig(
        iterations=catboost_config['iterations'],
        learning_rate=catboost_config['learning_rate'],
        depth=catboost_config['depth'],
        l2_leaf_reg=catboost_config['l2_leaf_reg'],
        auto_class_weights=catboost_config['auto_class_weights'],
        random_seed=catboost_config['random_seed'],
    )