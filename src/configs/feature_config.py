from dataclasses import dataclass
from src.utils.common import get_general_config
 
 
@dataclass
class FeatureConfig():
    output_path: str
    save_features: bool
    well_key_pattern: str
    test_size: float
    random_state: int
    valid_ranges: dict
    lithology_groups: dict
    shale_groups: list
    common: list
    by_type: dict
 
 
def get_feature_config() -> FeatureConfig:
    general_config = get_general_config()
    feature_config = general_config['features']
    return FeatureConfig(
        output_path=feature_config['output_path'],
        save_features=feature_config['save_features'],
        well_key_pattern=feature_config['well_key_pattern'],
        test_size=feature_config['test_size'],
        random_state=feature_config['random_state'],
        valid_ranges=feature_config['valid_ranges'],
        lithology_groups=feature_config['lithology_groups'],
        shale_groups=feature_config['shale_groups'],
        common=feature_config['common'],
        by_type=feature_config['by_type'],
    )