from dataclasses import dataclass
from src.utils.common import get_general_config
 
 
@dataclass
class EvaluationConfig():
    n_splits: int
    random_state: int
    primary_metric: str
    bootstrap_unit: str
    bootstrap_iterations: int
    confidence_level: float
 
 
def get_evaluation_config() -> EvaluationConfig:
    general_config = get_general_config()
    evaluation_config = general_config['evaluation']
    if evaluation_config['bootstrap_unit'] not in ('group', 'row'):
        raise ValueError(f"bootstrap_unit должен быть 'group' или 'row', получено: {evaluation_config['bootstrap_unit']}")
    return EvaluationConfig(
        n_splits=evaluation_config['n_splits'],
        random_state=evaluation_config['random_state'],
        primary_metric=evaluation_config['primary_metric'],
        bootstrap_unit=evaluation_config['bootstrap_unit'],
        bootstrap_iterations=evaluation_config['bootstrap_iterations'],
        confidence_level=evaluation_config['confidence_level'],
    )