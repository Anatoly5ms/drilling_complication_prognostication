from huggingface_hub import hf_hub_download
import pandas as pd
from src.configs.data_config import get_data_config
from src.utils.data_io import save_parquet

REPO_ID = 'bengsoon/volve_daily_drilling_report'
FILENAME = 'data/all-00000-of-00001.parquet'


def main() -> None:
    cfg = get_data_config()
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type='dataset')
    df = pd.read_parquet(cached_path)
    save_parquet(df, cfg.raw_ddr_path)  # путь из config.yaml: static/raw/volve_ddr.parquet
    print(f'Сохранено рапортов: {len(df)} -> {cfg.raw_ddr_path}')


if __name__ == '__main__':
    main()