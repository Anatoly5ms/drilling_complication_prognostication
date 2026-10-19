from dataclasses import dataclass
from src.utils.common import get_general_config
 
 
@dataclass
class DataConfig():
    raw_ddr_path: str
    interim_output_path: str
    save_interim: bool
    key_columns: list
    nested_tables: dict
    numeric_parse_threshold: float
    force_string_columns: list
    null_values: list
    datetime_prefixes: list
    datetime_columns: list
    deduplicate_reports: bool
 
 
def get_data_config() -> DataConfig:
    general_config = get_general_config()
    data_config = general_config['data']
    return DataConfig(
        raw_ddr_path=data_config['raw_ddr_path'],
        interim_output_path=data_config['interim_output_path'],
        save_interim=data_config['save_interim'],
        key_columns=data_config['key_columns'],
        nested_tables=data_config['nested_tables'],
        numeric_parse_threshold=data_config['numeric_parse_threshold'],
        force_string_columns=data_config['force_string_columns'],
        null_values=data_config['null_values'],
        datetime_prefixes=data_config['datetime_prefixes'],
        datetime_columns=data_config['datetime_columns'],
        deduplicate_reports=data_config['deduplicate_reports'],
    )