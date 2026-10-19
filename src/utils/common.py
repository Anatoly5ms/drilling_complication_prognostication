from yaml import safe_load
from pathlib import Path


def get_general_config() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    if not config_path.exists():
        raise ValueError('config.yaml is not found')
    with open(Path(config_path)) as yaml_file:
        return safe_load(yaml_file)
    

if __name__ == '__main__':
    yaml_content = get_general_config()