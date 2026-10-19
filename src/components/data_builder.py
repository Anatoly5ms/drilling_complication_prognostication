import numpy as np
import pandas as pd
from src.configs.data_config import DataConfig
from src.utils.data_io import load_parquet, save_tables


class DataBuilder():
    """
    Читает сырые суточные рапорты бурения (DDR) Volve и разворачивает вложенные поля WITSML в набор плоских таблиц, связанных ключом report_id:
        reports        — шапка рапорта (одна строка на рапорт)
        wellbore_info  — сведения о стволе
        status         — состояние на конец суток (глубина, диаметр, ROP, прочность пород и т.д.)
        activity       — операции за сутки (коды, глубина, комментарии) -> источник меток для Labeler
        fluid          — параметры бурового раствора
        lithology      — литология по интервалам глубин
        pore_pressure  — поровое давление
        survey         — инклинометрия
    Компонента ничего не знает про осложнения и признаки: только приводит данные к чистому табличному виду.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config
        self.tables = None
        self.quality_report = None

    def build_tables(self) -> dict[str, pd.DataFrame]:
        '''Метод читает сырой parquet, разворачивает вложенные поля в плоские таблицы, приводит типы, печатает отчёт о качестве и сохраняет таблицы в interim'''
        cfg = self.data_config

        raw_df = load_parquet(cfg.raw_ddr_path)
        reports = self._prepare_reports(raw_df)

        tables = {'reports': self._drop_nested_columns(reports.drop(columns=list(cfg.nested_tables), errors='ignore'))}
        for source_column, table_name in cfg.nested_tables.items():
            if source_column not in reports.columns:
                print(f'Поле {source_column} отсутствует в данных, таблица {table_name} пропущена')
                continue
            tables[table_name] = self._flatten_nested(reports, source_column)

        self.tables = {name: self._cast_types(df) for name, df in tables.items()}
        self.quality_report = self._build_quality_report(self.tables, raw_df_rows=len(raw_df))

        if cfg.save_interim:
            save_tables(self.tables, cfg.interim_output_path)

        return self.tables

    def _prepare_reports(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер переводит даты рапорта в UTC, удаляет дублирующиеся версии рапортов за один период (оставляет самую позднюю по createDate) и присваивает каждому рапорту ключ report_id'''
        cfg = self.data_config
        df = raw_df.copy()

        for column in ['dTimStart', 'dTimEnd', 'createDate']: # даты нужны раньше остальных типов, по ним сортируем и чистим дубликаты версий рапорта
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)

        if cfg.deduplicate_reports:
            rows_before = len(df)
            df = (df.sort_values('createDate', na_position='first').drop_duplicates(subset=['nameWellbore', 'dTimStart', 'dTimEnd'], keep='last'))
            removed = rows_before - len(df)
            if removed:
                print(f'Удалено дублирующихся версий рапортов: {removed}')

        df = df.sort_values(['nameWellbore', 'dTimStart']).reset_index(drop=True)
        df.insert(0, 'report_id', np.arange(len(df)))
        return df

    def _flatten_nested(self, reports: pd.DataFrame, source_column: str) -> pd.DataFrame:
        '''Этот хелпер разворачивает одно вложенное поле рапорта (список записей) в плоскую таблицу: одна запись — одна строка, к каждой строке приклеиваются ключи рапорта'''
        keys = ['report_id'] + self.data_config.key_columns + ['dTimStart', 'dTimEnd']
        keys = [k for k in keys if k in reports.columns]

        exploded = reports[keys + [source_column]].explode(source_column)
        exploded = exploded[exploded[source_column].apply(lambda x: isinstance(x, dict))]
        if exploded.empty:
            return pd.DataFrame(columns=keys)

        fields = pd.json_normalize(exploded[source_column].tolist())
        key_frame = exploded[keys].reset_index(drop=True).rename(columns={'dTimStart': 'report_dTimStart', 'dTimEnd': 'report_dTimEnd'}) # даты рапорта переименовываем, чтобы не путать с датами внутри вложенной записи (dTim и т.д.)
        flat = pd.concat([key_frame, fields.reset_index(drop=True)], axis=1)
        return self._drop_nested_columns(flat)

    def _drop_nested_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер удаляет колонки, в которых остались вложенные списки или словари второго уровня (rigAlias и т.п.)'''
        nested = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any()] # вложенные списки второго уровня (rigAlias и т.п.) для задачи НЕ нужны
        return df.drop(columns=nested)

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер приводит колонки к правильным типам: даты – в UTC; коды и текст – в строки; остальное – в числа, если распарсилось не меньше доли numeric_parse_threshold. Коды пропуска (-999.99 и т.п.) превращаются в пропуски'''
        cfg = self.data_config
        df = df.copy()

        for column in df.columns:
            if column == 'report_id' or pd.api.types.is_datetime64_any_dtype(df[column]):
                continue
            is_datetime = (column in cfg.datetime_columns or any(column.startswith(prefix) for prefix in cfg.datetime_prefixes))
            if is_datetime:
                df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)
                continue

            if column in cfg.force_string_columns or column in cfg.key_columns:
                df[column] = df[column].astype('string').str.strip()
                continue

            values = df[column].astype('string').str.strip().replace('', pd.NA) # пробуем число. Если распарсилось достаточно значений, колонка числовая
            parsed = pd.to_numeric(values, errors='coerce')
            non_null = values.notna().sum()
            parse_rate = parsed.notna().sum() / non_null if non_null else 0.0
            parsed = parsed.mask(parsed.isin(cfg.null_values))  # коды пропуска -> пропуск (после подсчёта доли: -999.99 — число, а не текст)
            df[column] = parsed.astype('float64') if parse_rate >= cfg.numeric_parse_threshold else values
        return df

    def _build_quality_report(self, tables: dict[str, pd.DataFrame], raw_df_rows: int) -> pd.DataFrame:
        '''Этот хелпер собирает отчёт о качестве по каждой колонке (тип, доля пропусков, число уникальных значений) и печатает строковые колонки, где часть значений похожа на числа. Возможно, с единицами измерения внутри'''
        cfg = self.data_config
        rows = []
        for table_name, df in tables.items():
            for column in df.columns:
                series = df[column]
                row = {
                    'table': table_name,
                    'column': column,
                    'dtype': str(series.dtype),
                    'missing_share': round(series.isna().mean(), 3),
                    'n_unique': series.nunique(dropna=True),
                    'partial_numeric_share': None,
                }
                if series.dtype == 'string' and column not in cfg.force_string_columns: # строковая колонка, где часть значений является числом, а другая – строко (например, число с единицей измерения типа "1.25 sg")
                    non_null = series.notna().sum()
                    if non_null:
                        share = pd.to_numeric(series, errors='coerce').notna().sum() / non_null
                        row['partial_numeric_share'] = round(share, 3) if share > 0 else None
                rows.append(row)

        report = pd.DataFrame(rows)
        summary = pd.DataFrame(
            [
                {
                    'table': name,
                    'rows': len(df),
                    'wellbores': df['nameWellbore'].nunique() if 'nameWellbore' in df.columns else None,
                    'reports_covered': df['report_id'].nunique() if 'report_id' in df.columns else None,
                } for name, df in tables.items()
            ]
        )
        # print(f'Сырых рапортов: {raw_df_rows}')
        # print(summary.to_string(index=False))

        suspicious = report[report['partial_numeric_share'].notna()]
        if not suspicious.empty:
            print('\nСтроковые колонки с частично числовыми значениями (проверить единицы измерения):')
            print(suspicious[['table', 'column', 'partial_numeric_share']].to_string(index=False))
        return report