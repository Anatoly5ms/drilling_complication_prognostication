from dataclasses import dataclass
from src.utils.common import get_general_config


@dataclass
class LogisticRegressionConfig():
    C: float
    class_weight: str | None
    max_iter: int
    add_missing_indicator: bool


def get_logistic_regression_config() -> LogisticRegressionConfig:
    general_config = get_general_config()
    lr_config = general_config['logistic_regression']
    return LogisticRegressionConfig(
        C=lr_config['C'],
        class_weight=lr_config['class_weight'],
        max_iter=lr_config['max_iter'],
        add_missing_indicator=lr_config['add_missing_indicator'],
    )