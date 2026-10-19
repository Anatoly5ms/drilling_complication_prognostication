import re
import numpy as np
import pandas as pd
from src.configs.feature_config import FeatureConfig
from src.configs.target_config import TargetConfig
from src.utils.data_io import save_parquet
from src.utils.splitting import assign_holdout_split
 
 
class FeatureBuilder():
    """
    Строит признаки для каждого интервала глубины из сетки Labeler и собирает датасет для моделей.
    Главное правило - никакой утечки из будущего: параметры раствора, давления и состояние скважины
    берутся только из записей, сделанных до момента входа долота в интервал (entered_at). Прогноз риска
    для интервала должен быть доступен до того, как его начали бурить.
    Сначала строится полный набор признаков (одинаковый для всех типов осложнений), затем из него
    выбираются признаки по конфигу: features.common + features.by_type[complication_type].
    """
    
    ID_COLUMNS = ['interval_id', 'well_key', 'nameWell', 'nameWellbore', 'md_top', 'md_bottom', 'entered_at', 'drilled_at']
 
    def __init__(self, feature_config: FeatureConfig, target_config: TargetConfig,
                 tables: dict[str, pd.DataFrame], intervals: pd.DataFrame) -> None:
        self.feature_config = feature_config
        self.target_config = target_config
        self.tables = tables
        self.intervals = intervals
        self.all_features = None
        self.dataset = None
        self.feature_report = None
        self.feature_columns = None
 
    def build_features(self) -> pd.DataFrame:
        cfg = self.feature_config
        complication_type = self.target_config.complication_type
        label_column = f'label_{complication_type}'
 
        df = self.intervals.copy()
        df['md_mid'] = (df['md_top'] + df['md_bottom']) / 2
        df['well_key'] = self._build_well_key(df)
        df = self._add_geometry(df)
        df = self._add_status_features(df)
        df = self._add_fluid_features(df)
        df = self._add_pore_pressure(df)
        df = self._add_lithology(df)
        df = self._apply_valid_ranges(df)
        df = self._add_derived_features(df)
        self.all_features = df
 
        feature_columns = list(dict.fromkeys(cfg.common + cfg.by_type.get(complication_type, [])))
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f'Признаки из конфига не построены: {missing}')
 
        id_columns = [c for c in self.ID_COLUMNS if c not in feature_columns] # md_top одновременно идентификатор интервала и признак - колонку не дублируем
        dataset = df[id_columns + feature_columns + [label_column]].copy()
        dataset = dataset.rename(columns={label_column: 'label'})
        dataset['split'] = assign_holdout_split(dataset, 'well_key', 'label', cfg.test_size, cfg.random_state)
        self.dataset = dataset
        self.feature_columns = feature_columns
        # self.feature_report = self._build_feature_report(dataset, feature_columns)
 
        if cfg.save_features:
            save_parquet(dataset, f'{cfg.output_path}/features_{complication_type}.parquet')
        return dataset
 
    @staticmethod
    def _asof_by_time(df: pd.DataFrame, source: pd.DataFrame, columns: dict) -> pd.DataFrame:
        '''Этот хелпер приклеивает к интервалам последнюю запись источника, сделанную не позже entered_at (без утечки из будущего)'''
        source = source.dropna(subset=['record_time'])[['nameWellbore', 'record_time'] + list(columns)].rename(columns=columns)
        source['nameWellbore'] = source['nameWellbore'].astype(object) # merge_asof требует одинаковых типов ключей: приводим ствол к строке, время к одной точности
        source['record_time'] = source['record_time'].astype('datetime64[ns, UTC]')
        source = source.sort_values('record_time')
 
        df = df.copy()
        df['nameWellbore'] = df['nameWellbore'].astype(object)
        df['entered_at'] = pd.to_datetime(df['entered_at'], utc=True).astype('datetime64[ns, UTC]')
        has_time = df['entered_at'].notna()
        merged = pd.merge_asof(
            df[has_time].sort_values('entered_at'), source,
            left_on='entered_at', right_on='record_time', by='nameWellbore', direction='backward',
        ).drop(columns='record_time')
        return pd.concat([merged, df[~has_time]], ignore_index=True)
 
    @staticmethod
    def _record_time(table: pd.DataFrame) -> pd.Series:
        '''Этот хелпер возвращает время записи: собственное (dTim), а если его нет - конец суток рапорта'''
        own = table['dTim'] if 'dTim' in table.columns else pd.Series(pd.NaT, index=table.index)
        return own.fillna(table['report_dTimEnd'])
 
    def _add_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер интерполирует по инклинометрии вертикальную глубину и зенитный угол в середине и на границах интервала.
        Траектория - это геометрия ствола, а не измерение процесса, поэтому здесь используются все замеры ствола'''
        survey = self.tables['survey'].dropna(subset=['md'])
        df = df.copy()
        df['tvd'] = np.nan
        df['incl'] = np.nan
        df['incl_change'] = np.nan
 
        for wellbore, stations in survey.groupby('nameWellbore'):
            stations = stations.sort_values('md').drop_duplicates('md')
            rows = df['nameWellbore'] == wellbore
            md_min, md_max = stations['md'].min(), stations['md'].max()
 
            def interp(md, column):
                values = np.interp(md, stations['md'], stations[column])
                return np.where((md >= md_min) & (md <= md_max), values, np.nan)  # без экстраполяции за пределы замеров
 
            if stations['tvd'].notna().any():
                df.loc[rows, 'tvd'] = interp(df.loc[rows, 'md_mid'], 'tvd')
            if stations['incl'].notna().any():
                df.loc[rows, 'incl'] = interp(df.loc[rows, 'md_mid'], 'incl')
                df.loc[rows, 'incl_change'] = np.abs(interp(df.loc[rows, 'md_bottom'], 'incl') - interp(df.loc[rows, 'md_top'], 'incl'))
        return df
 
    def _add_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер берёт из состояния скважины на последний рапорт до entered_at: диаметр ствола, башмак последней колонны и FIT/LOT'''
        status = self.tables['status'].copy()
        status['record_time'] = self._record_time(status)
        columns = {c: name for c, name in [('diaHole', 'hole_diameter'), ('mdCsgLast', 'casing_shoe_md'), ('strengthForm', 'fit_emw')] if c in status.columns}
        df = self._asof_by_time(df, status, columns)
        if 'casing_shoe_md' in df.columns: # у бокового ствола в рапорте часто указан башмак колонны основного ствола, который глубже окна зарезки. Тогда открытый ствол начинается от точки зарезки – первой пробуренной глубины этого ствола
            kickoff_md = df.groupby('nameWellbore')['md_top'].transform('min')
            effective_shoe = df['casing_shoe_md'].where(df['casing_shoe_md'] <= df['md_top'], kickoff_md)
            effective_shoe = effective_shoe.where(df['casing_shoe_md'].notna())  # нет данных о башмаке – нет и признака
            df['open_hole_length'] = df['md_mid'] - effective_shoe
        else:
            df['open_hole_length'] = np.nan
        return df
 
    def _add_fluid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер берёт последний замер бурового раствора до entered_at: плотность, пластическую вязкость, ДНС, тип'''
        fluid = self.tables['fluid'].copy()
        fluid['record_time'] = self._record_time(fluid)
        columns = {c: name for c, name in [('density', 'mud_density'), ('pv', 'mud_pv'), ('yp', 'mud_yp'), ('mudClass', 'mud_class')] if c in fluid.columns}
        return self._asof_by_time(df, fluid, columns)
 
    def _add_pore_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер берёт последнюю оценку порового давления (в эквиваленте плотности) до entered_at'''
        pore = self.tables['pore_pressure'].copy()
        pore['record_time'] = self._record_time(pore)
        return self._asof_by_time(df, pore, {'equivalentMudWeight': 'pore_pressure_emw'})
 
    def _build_well_key(self, df: pd.DataFrame) -> pd.Series:
        '''Этот хелпер выделяет ключ скважины из названия ствола (NO 15/9-19 BT2 -> NO 15/9-19) для разбиения train/test'''
        key = df['nameWellbore'].astype(str).str.extract(self.feature_config.well_key_pattern, expand=False)
        return key.fillna(df['nameWell'].astype(str))  # если шаблон не подошёл, берём nameWell как есть
 
    def _classify_lithology(self, description: str) -> str:
        '''Этот хелпер относит текстовое описание породы к группе: побеждает группа, чей шаблон встречается в тексте раньше всех'''
        best_group, best_position = 'other', None
        for group, patterns in self._lithology_patterns.items():
            for pattern in patterns:
                match = pattern.search(description)
                if match and (best_position is None or match.start() < best_position):
                    best_group, best_position = group, match.start()
        return best_group
 
    def _add_lithology(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер определяет литологию интервала по пересечению с описанием разреза:
        преобладающую группу пород (lithology_group) и долю глинистых пород (shale_share)'''
        cfg = self.feature_config
        self._lithology_patterns = {group: [re.compile(p, re.IGNORECASE) for p in patterns] for group, patterns in cfg.lithology_groups.items()}
        lith = self.tables['lithology'].dropna(subset=['mdTop', 'mdBottom', 'lithology']).copy()
        lith['group'] = lith['lithology'].astype(str).apply(self._classify_lithology)
 
        df = df.copy()
        df['lithology_group'] = pd.Series(pd.NA, index=df.index, dtype='string')
        df['shale_share'] = np.nan
 
        for wellbore, layers in lith.groupby('nameWellbore'):
            for idx in df.index[df['nameWellbore'] == wellbore]:
                top, bottom = df.at[idx, 'md_top'], df.at[idx, 'md_bottom']
                overlap = (np.minimum(layers['mdBottom'], bottom) - np.maximum(layers['mdTop'], top)).clip(lower=0)
                if overlap.sum() == 0:
                    continue
                by_group = overlap.groupby(layers['group']).sum()
                df.at[idx, 'lithology_group'] = by_group.idxmax()
                df.at[idx, 'shale_share'] = by_group[by_group.index.isin(cfg.shale_groups)].sum() / by_group.sum()
        return df
 
    def _apply_valid_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер превращает значения вне допустимых диапазонов в пропуски (ошибки данных, FIT = 0 и т.п.)'''
        df = df.copy()
        for column, (low, high) in self.feature_config.valid_ranges.items():
            if column not in df.columns:
                continue
            invalid = df[column].notna() & ~df[column].between(low, high)
            if invalid.any():
                print(f'{column}: вне диапазона [{low}, {high}] -> пропуск: {int(invalid.sum())} значений из {len(df)}')
                df.loc[invalid, column] = np.nan
        return df
 
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер считает производные признаки из уже построенных'''
        df = df.copy()
        df['fracture_margin'] = df['fit_emw'] - df['mud_density'] #  запас до гидроразрыва: насколько плотность раствора ниже прочности пород по FIT/LOT (ECD в данных нет, плотность приближенная)
        df['overbalance'] = df['mud_density'] - df['pore_pressure_emw'] # репрессия на пласт: превышение плотности раствора над поровым давлением
        return df
 
    def _build_feature_report(self, dataset: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        '''Этот хелпер печатает заполненность и диапазоны признаков, а также состав train/test'''
        rows = []
        for column in feature_columns:
            series = dataset[column]
            numeric = pd.api.types.is_numeric_dtype(series)
            rows.append({
                'feature': column,
                'missing_share': round(series.isna().mean(), 3),
                'min': round(series.min(), 3) if numeric and series.notna().any() else None,
                'median': round(series.median(), 3) if numeric and series.notna().any() else None,
                'max': round(series.max(), 3) if numeric and series.notna().any() else None,
                'n_unique': series.nunique(dropna=True),
            })
        report = pd.DataFrame(rows)
        split_summary = dataset.groupby('split').agg(intervals=('label', 'size'), positives=('label', 'sum'), wells=('well_key', 'nunique'),
        )
        print(f'\nПризнаки для {self.target_config.complication_type}:')
        print(report.to_string(index=False))
        print('\nРазбиение train/test по скважинам:')
        print(split_summary.to_string())
        return report