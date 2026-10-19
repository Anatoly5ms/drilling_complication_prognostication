from dataclasses import dataclass
from src.utils.common import get_general_config
 
 
@dataclass
class TargetConfig():
    complication_type: str
    min_positive_events: dict
 
 
def get_target_config() -> TargetConfig:
    general_config = get_general_config()
    target_config = general_config['target']
    supported_types = list(general_config['labeler']['complication_rules'])
    complication_type = target_config['complication_type']
    if complication_type not in supported_types:
        raise ValueError(f'Неизвестный тип осложнения: {complication_type}. Допустимые: {supported_types}')
    if complication_type not in target_config['min_positive_events']:
        raise ValueError(f'Для {complication_type} не задан min_positive_events')
    return TargetConfig(
        complication_type=complication_type,
        min_positive_events=target_config['min_positive_events'],
    )