from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_parquet(path: str) -> pd.DataFrame:
    full_path = resolve_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f'Файл не найден: {full_path}')
    return pd.read_parquet(full_path)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    full_path = resolve_path(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(full_path, index=False)


def save_tables(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    for table_name, df in tables.items():
        save_parquet(df, str(Path(output_dir) / f'{table_name}.parquet'))


def load_tables(table_names: list, input_dir: str) -> dict[str, pd.DataFrame]:
    return {name: load_parquet(str(Path(input_dir) / f'{name}.parquet')) for name in table_names}