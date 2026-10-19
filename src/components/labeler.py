import re
import numpy as np
import pandas as pd
from src.configs.labeler_config import LabelerConfig
from src.configs.target_config import TargetConfig
from src.utils.data_io import save_tables


class Labeler():
    """
    Размечает осложнения по журналу операций суточных рапортов и строит сетку интервалов глубины.
    Шаги:
        1. Отбирает операции бурения и НПВ (без заканчивания, КРС и ликвидации) и считает глубину забоя на момент каждой операции.
        2. Для каждого типа осложнения находит операции по коду и/или тексту комментария (сильные и слабые шаблоны).
        3. Склеивает соседние по времени и глубине операции в события (одно поглощение = несколько строк журнала).
        4. Строит сетку пробуренных интервалов по каждому стволу: когда долото вошло в интервал (entered_at - момент прогноза)
           и когда прошло его (drilled_at).
        5. Ставит метки на интервалы (мультиметочно: по колонке на каждый тип) с допуском по глубине только вверх:
           поглощение происходит в уже пробуренном стволе, а интервала ниже забоя в момент события ещё не было.
    Метки строятся сразу для всех типов (это дёшево), а target.complication_type используется для проверки, хватает ли событий для обучения.
    Событие со слабым совпадением помечается match_strength='weak': сейчас оно положительное при weak_as_positive=true,
    а в будущем такие события будет проверять LLM.
    """

    def __init__(self, labeler_config: LabelerConfig, target_config: TargetConfig, tables: dict[str, pd.DataFrame]) -> None:
        self.labeler_config = labeler_config
        self.target_config = target_config
        self.tables = tables
        self.events = None
        self.intervals = None
        self.label_report = None

    def build_labels(self) -> dict[str, pd.DataFrame]:
        cfg = self.labeler_config

        activity = self._prepare_activity(self.tables['activity'])
        activity = self._add_event_depth(activity)

        events = []
        for complication_type, rules in cfg.complication_rules.items():
            matched = self._match_operations(activity, rules)
            events.append(self._group_events(matched, complication_type))
        events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()

        intervals = self._build_interval_grid(activity)
        self.events = self._flag_events_in_drilled_range(events, intervals)
        self.intervals = self._assign_labels(intervals, self.events)
        self.label_report = self._build_label_report(self.events, self.intervals)

        if cfg.save_labels:
            save_tables({'events': self.events, 'intervals': self.intervals, 'label_report': self.label_report}, cfg.output_path)
        return {'events': self.events, 'intervals': self.intervals}

    def has_enough_events(self, complication_type: str | None = None) -> bool:
        '''Метод проверяет, хватает ли событий выбранного типа для обучения (сравнивает с порогом из конфига)'''
        complication_type = complication_type or self.target_config.complication_type
        threshold = self.target_config.min_positive_events[complication_type]
        row = self.label_report.loc[self.label_report['complication_type'] == complication_type]
        n_events = int(row['events_in_drilled_range'].iloc[0]) if not row.empty else 0
        if n_events < threshold:
            print(f'{complication_type}: событий в пробуренных интервалах {n_events} < порога {threshold}. '
                  f'Обучение не запускается.')
            return False
        return True

    def _prepare_activity(self, activity: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер оставляет только операции бурения и НПВ + сортирует по стволу и времени'''
        cfg = self.labeler_config
        df = activity.copy()
        df['code_phase'] = df['proprietaryCode'].str.split(' -- ').str[0].str.strip()
        df = df[df['code_phase'].isin(cfg.relevant_phases)]
        df['md'] = df['md'].where(df['md'] >= cfg.min_valid_md) # глубины меньше min_valid_md (в т.ч. 0, что означает, что инструмент на поверхности) считаем отсутствием данных
        return df.sort_values(['nameWellbore', 'dTimStart']).reset_index(drop=True)

    def _add_event_depth(self, activity: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер считает глубину забоя на момент каждой операции и выбирает глубину для событий (md_depth)'''
        cfg = self.labeler_config
        df = activity.copy()

        # забой = максимальная глубина, пробуренная к началу операции (по операциям бурения)
        drill = df[df['proprietaryCode'].isin(cfg.drilling_codes) & df['md'].notna()][['nameWellbore', 'dTimEnd', 'md']]
        drill = drill.sort_values('dTimEnd').rename(columns={'dTimEnd': 'drill_time', 'md': 'drilled_md'})
        drill['hole_depth'] = drill.groupby('nameWellbore')['drilled_md'].cummax()

        has_time = df['dTimStart'].notna()
        timed = pd.merge_asof(
            df[has_time].sort_values('dTimStart'), drill[['nameWellbore', 'drill_time', 'hole_depth']],
            left_on='dTimStart', right_on='drill_time', by='nameWellbore', direction='backward',
        ).drop(columns='drill_time')
        df = pd.concat([timed, df[~has_time]], ignore_index=True)

        # если в самой операции долото глубже прежнего забоя (бурение), то забой – это глубина долота
        hole_depth = df[['hole_depth', 'md']].max(axis=1, skipna=True)
        df['md_depth'] = hole_depth.fillna(df['md']) if cfg.event_depth_source == 'hole_depth' else df['md']
        return df.sort_values(['nameWellbore', 'dTimStart']).reset_index(drop=True)
 
    def _match_operations(self, activity: pd.DataFrame, rules: dict) -> pd.DataFrame:
        '''Этот хелпер находит операции, относящиеся к осложнению: по коду операции и/или по тексту комментария, помечает источник (code, text, code+text) и силу совпадения (strong, weak)'''
        strong = [re.compile(p, re.IGNORECASE) for p in rules.get('strong_patterns', [])]
        weak = [re.compile(p, re.IGNORECASE) for p in rules.get('weak_patterns', [])]
        exclude = [re.compile(p, re.IGNORECASE) for p in rules.get('exclude_patterns', [])]

        text_strength = activity['comments'].apply(lambda c: self._text_match_strength(c, strong, weak, exclude))
        by_text = text_strength.notna()
 
        by_code = activity['proprietaryCode'].isin(rules.get('codes', []))
        if rules.get('code_requires_text', False):
            by_code = by_code & by_text  # коды в данных ненадёжны: засчитываем только с подтверждением текстом
 
        matched = activity[by_code | by_text].copy()
        code_hit, text_hit = by_code[matched.index], by_text[matched.index]
        matched['source'] = np.select([code_hit & text_hit, code_hit], ['code+text', 'code'], default='text')
        matched['match_strength'] = np.where(code_hit | (text_strength[matched.index] == 'strong'), 'strong', 'weak') # код операции считаем явным признаком, поэтому он даёт сильное совпадение
        return matched
 
    @staticmethod
    def _text_match_strength(comment, strong: list, weak: list, exclude: list) -> str | None:
        '''Этот хелпер проверяет комментарий по фразам и возвращает силу совпадения: strong, weak или None'''
        if pd.isna(comment):
            return None
        found_weak = False
        for phrase in re.split(r'[.;\n]+|,\s+', str(comment)): # выделяем фразы по точкам, точкам с запятой, переносам и запятой с пробелом ("1,08 m3" при этом не делится). "Losses of 6 m3/hr at 500 lpm, no losses at 250 lpm" – первая фраза засчитывается, вторая отбрасывается
            if any(p.search(phrase) for p in exclude):
                continue
            if any(p.search(phrase) for p in strong):
                return 'strong'
            if any(p.search(phrase) for p in weak):
                found_weak = True
        return 'weak' if found_weak else None
 
    def _group_events(self, matched: pd.DataFrame, complication_type: str) -> pd.DataFrame:
        '''Этот хелпер склеивает соседние по времени и глубине операции одного ствола в одно событие осложнения'''
        cfg = self.labeler_config
        columns = ['complication_type', 'event_id', 'nameWell', 'nameWellbore', 'dTimStart', 'dTimEnd',
                   'md_event', 'md_bit', 'md_min', 'md_max', 'n_operations', 'sources', 'match_strength',
                   'is_positive', 'comments_sample']
        if matched.empty:
            return pd.DataFrame(columns=columns)
 
        df = matched.sort_values(['nameWellbore', 'dTimStart']).copy()
        prev_end = df.groupby('nameWellbore')['dTimEnd'].shift()
        prev_depth = df.groupby('nameWellbore')['md_depth'].shift()
        gap_hours = (df['dTimStart'] - prev_end).dt.total_seconds() / 3600
        depth_jump = (df['md_depth'] - prev_depth).abs()
 
        new_event = (prev_end.isna() | (gap_hours > cfg.event_gap_hours) | (depth_jump > cfg.event_max_depth_gap).fillna(False)) # новое событие – это новый ствол, большой разрыв по времени или скачок глубины (по md_depth, а не по долоту)
        df['event_id'] = new_event.cumsum()
 
        events = df.groupby('event_id').agg(
            nameWell=('nameWell', 'first'),
            nameWellbore=('nameWellbore', 'first'),
            dTimStart=('dTimStart', 'min'),
            dTimEnd=('dTimEnd', 'max'),
            md_event=('md_depth', 'first'),  # глубина события по выбранному источнику (event_depth_source)
            md_bit=('md', 'first'),  # глубина долота в первой операции - для справки
            md_min=('md_depth', 'min'),
            md_max=('md_depth', 'max'),
            n_operations=('md_depth', 'size'),
            sources=('source', lambda s: ','.join(sorted(set(s)))),
            match_strength=('match_strength', lambda s: 'strong' if (s == 'strong').any() else 'weak'),
            comments_sample=('comments', lambda s: ' | '.join(s.dropna().astype(str))[:500]),
        ).reset_index()
        events['is_positive'] = (events['match_strength'] == 'strong') | cfg.weak_as_positive
        events.insert(0, 'complication_type', complication_type)
        events['event_id'] = complication_type + '_' + events['event_id'].astype(str)
        return events[columns]
 
    def _build_interval_grid(self, activity: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер строит сетку интервалов по 10 м для каждого ствола и два момента времени:
        entered_at - начало операции, в которой долото вошло в интервал (момент прогноза: признаки берутся только до него),
        drilled_at - окончание операции, в которой долото прошло низ интервала (для справки)'''
        cfg = self.labeler_config
        step = cfg.interval_length
        drill = activity[activity['proprietaryCode'].isin(cfg.drilling_codes) & activity['md'].notna()]
 
        grids = []
        for wellbore, ops in drill.groupby('nameWellbore'):
            ops = ops.sort_values('dTimEnd')
            top = np.floor(ops['md'].min() / step) * step
            bottom = (np.floor(ops['md'].max() / step) + 1) * step # включаем интервал, в котором находится максимальная глубина (забой)
            md_top = np.arange(top, bottom, step)
 
            running_max = ops['md'].cummax().to_numpy() # накопленный максимум глубины бурения = забой после каждой операции
            last = len(ops) - 1
            # вход в интервал: первая операция, после которой забой достиг кровли интервала; берём её НАЧАЛО -
            # в этот момент долото ещё не вошло в интервал, поэтому всё, что известно к нему, известно до бурения интервала
            entered_idx = np.searchsorted(running_max, np.minimum(md_top, running_max[-1]), side='left')
            entered_at = ops['dTimStart'].to_numpy()[np.minimum(entered_idx, last)]
            # проход интервала: первая операция, после которой забой достиг подошвы интервала; берём её окончание
            drilled_idx = np.searchsorted(running_max, np.minimum(md_top + step, running_max[-1]), side='left')
            drilled_at = ops['dTimEnd'].to_numpy()[np.minimum(drilled_idx, last)]
 
            grids.append(pd.DataFrame({
                'nameWell': ops['nameWell'].iloc[0],
                'nameWellbore': wellbore,
                'md_top': md_top,
                'md_bottom': md_top + step,
                'entered_at': entered_at,
                'drilled_at': drilled_at,
            }))
 
        if not grids:
            return pd.DataFrame(columns=['nameWell', 'nameWellbore', 'md_top', 'md_bottom', 'entered_at', 'drilled_at'])
        intervals = pd.concat(grids, ignore_index=True)
        intervals.insert(0, 'interval_id', intervals['nameWellbore'] + '_' + intervals['md_top'].astype(int).astype(str))
        return intervals
 
    def _flag_events_in_drilled_range(self, events: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер отмечает, попадает ли глубина события в пробуренный диапазон ствола'''
        if events.empty:
            events['in_drilled_range'] = pd.Series(dtype=bool)
            return events
        ranges = intervals.groupby('nameWellbore').agg(grid_top=('md_top', 'min'), grid_bottom=('md_bottom', 'max'))
        events = events.merge(ranges, left_on='nameWellbore', right_index=True, how='left')
        events['in_drilled_range'] = (events['md_event'].notna() & (events['md_event'] >= events['grid_top']) & (events['md_event'] <= events['grid_bottom'])).fillna(False)
        return events.drop(columns=['grid_top', 'grid_bottom'])
 
    def _assign_labels(self, intervals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер ставит метки на интервал с положительным событием и на соседние интервалы (несимметричный допуск по глубине), по колонке на каждый тип'''
        cfg = self.labeler_config
        step = cfg.interval_length
        above, below = cfg.label_tolerance_above, cfg.label_tolerance_below
        intervals = intervals.copy()
 
        for complication_type in cfg.complication_rules:
            label_column = f'label_{complication_type}'
            intervals[label_column] = 0
            type_events = events[(events['complication_type'] == complication_type) & events['in_drilled_range'] & events['is_positive']]
 
            for _, event in type_events.iterrows():
                event_top = np.floor(event['md_event'] / step) * step
                in_wellbore = intervals['nameWellbore'] == event['nameWellbore']
                near = intervals['md_top'].between(event_top - above * step, event_top + below * step)
                intervals.loc[in_wellbore & near, label_column] = 1
        return intervals
 
    def _build_label_report(self, events: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер считает и печатает сводку по каждому типу осложнения'''
        rows = []
        for complication_type in self.labeler_config.complication_rules:
            type_events = events[events['complication_type'] == complication_type]
            positive = type_events[type_events['is_positive'] & type_events['in_drilled_range']] if len(type_events) else type_events
            label = intervals[f'label_{complication_type}']
            rows.append(
                {
                    'complication_type': complication_type,
                    'events_total': len(type_events),
                    'events_in_drilled_range': len(positive),  # положительные события в пробуренном диапазоне - именно их сравниваем с порогом
                    'events_strong': int((positive['match_strength'] == 'strong').sum()) if len(positive) else 0,
                    'events_weak': int((positive['match_strength'] == 'weak').sum()) if len(positive) else 0,
                    'events_by_code': int(positive['sources'].str.contains('code').sum()) if len(positive) else 0,
                    'events_text_only': int((positive['sources'] == 'text').sum()) if len(positive) else 0,
                    'positive_intervals': int(label.sum()),
                    'prevalence': round(label.mean(), 4) if len(label) else 0.0,
                    'wellbores_with_events': positive['nameWellbore'].nunique() if len(positive) else 0,
                }
            )
        report = pd.DataFrame(rows)
        # print(f'Интервалов: {len(intervals)}, стволов: {intervals["nameWellbore"].nunique()}')
        # print(report.to_string(index=False))
        return report