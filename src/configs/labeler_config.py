from dataclasses import dataclass
from src.utils.common import get_general_config


@dataclass
class LabelerConfig():
    output_path: str
    save_labels: bool
    interval_length: float
    label_tolerance_above: int
    label_tolerance_below: int
    event_gap_hours: float
    event_max_depth_gap: float
    relevant_phases: list
    min_valid_md: float
    drilling_codes: list
    event_depth_source: str
    weak_as_positive: bool
    complication_rules: dict


def get_labeler_config() -> LabelerConfig:
    general_config = get_general_config()
    labeler_config = general_config['labeler']
    return LabelerConfig(
        output_path=labeler_config['output_path'],
        save_labels=labeler_config['save_labels'],
        interval_length=labeler_config['interval_length'],
        label_tolerance_above=labeler_config['label_tolerance_above'],
        label_tolerance_below=labeler_config['label_tolerance_below'],
        event_gap_hours=labeler_config['event_gap_hours'],
        event_max_depth_gap=labeler_config['event_max_depth_gap'],
        relevant_phases=labeler_config['relevant_phases'],
        min_valid_md=labeler_config['min_valid_md'],
        drilling_codes=labeler_config['drilling_codes'],
        event_depth_source=labeler_config['event_depth_source'],
        weak_as_positive=labeler_config['weak_as_positive'],
        complication_rules=labeler_config['complication_rules'],
    )