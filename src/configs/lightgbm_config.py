from dataclasses import dataclass
from src.utils.common import get_general_config


@dataclass
class LightGBMConfig():
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_lambda: float
    class_weight: str | None
    random_state: int


def get_lightgbm_config() -> LightGBMConfig:
    general_config = get_general_config()
    lgbm_config = general_config['lightgbm']
    return LightGBMConfig(
        n_estimators=lgbm_config['n_estimators'],
        learning_rate=lgbm_config['learning_rate'],
        num_leaves=lgbm_config['num_leaves'],
        min_child_samples=lgbm_config['min_child_samples'],
        subsample=lgbm_config['subsample'],
        subsample_freq=lgbm_config['subsample_freq'],
        colsample_bytree=lgbm_config['colsample_bytree'],
        reg_lambda=lgbm_config['reg_lambda'],
        class_weight=lgbm_config['class_weight'],
        random_state=lgbm_config['random_state'],
    )