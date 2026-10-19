import numpy as np
import pandas as pd

from src.pipeline.training_pipeline import TrainingPipeline


def match_intervals_to_events(dataset: pd.DataFrame, events: pd.DataFrame, step: float, above: int, below: int) -> pd.DataFrame:
    '''Для каждого положительного интервала находит событие, которое дало ему метку, и положение интервала относительно глубины события'''
    positives = dataset[dataset['label'] == 1].copy()
    rows = []
    for _, event in events.iterrows():
        event_top = np.floor(event['md_event'] / step) * step
        near = positives[(positives['nameWellbore'] == event['nameWellbore'])
                         & positives['md_top'].between(event_top - above * step, event_top + below * step)]
        for idx, interval in near.iterrows():
            offset = int(round((interval['md_top'] - event_top) / step))
            rows.append({'row': idx, 'event_id': event['event_id'], 'offset': offset,
                         'event_start': event['dTimStart']})
    matched = pd.DataFrame(rows).drop_duplicates('row')  # интервал мог попасть под два события - берём первое
    return positives.join(matched.set_index('row'), how='inner')


def add_fluid_record_time(intervals: pd.DataFrame, fluid: pd.DataFrame) -> pd.DataFrame:
    '''Находит время замера раствора, из которого Feature builder взял признаки интервала (последний замер до entered_at)'''
    fluid = fluid.dropna(subset=['density']).copy()
    fluid['fluid_time'] = fluid['dTim'].fillna(fluid['report_dTimEnd']).astype('datetime64[ns, UTC]')
    fluid['nameWellbore'] = fluid['nameWellbore'].astype(object)
    left = intervals.copy()
    left['nameWellbore'] = left['nameWellbore'].astype(object)
    left['entered_at'] = pd.to_datetime(left['entered_at'], utc=True).astype('datetime64[ns, UTC]')
    left = left.reset_index().sort_values('entered_at')
    merged = pd.merge_asof(left, fluid[['nameWellbore', 'fluid_time']].sort_values('fluid_time'),
                           left_on='entered_at', right_on='fluid_time', by='nameWellbore', direction='backward')
    return merged.set_index('index').sort_index()


def mud_density_change(fluid: pd.DataFrame, events: pd.DataFrame, window_hours: float = 48) -> pd.DataFrame:
    '''Сравнивает плотность раствора до начала поглощения и в течение window_hours после его окончания'''
    fluid = fluid.dropna(subset=['density']).copy()
    fluid['time'] = fluid['dTim'].fillna(fluid['report_dTimEnd'])
    rows = []
    for _, event in events.iterrows():
        wellbore = fluid[fluid['nameWellbore'] == event['nameWellbore']]
        before = wellbore[wellbore['time'] <= event['dTimStart']].sort_values('time')
        after = wellbore[(wellbore['time'] >= event['dTimEnd'])
                         & (wellbore['time'] <= event['dTimEnd'] + pd.Timedelta(hours=window_hours))].sort_values('time')
        if before.empty or after.empty:
            continue
        rows.append({'event_id': event['event_id'], 'density_before': before['density'].iloc[-1],
                     'density_after': after['density'].iloc[0]})
    result = pd.DataFrame(rows)
    if not result.empty:
        result['change'] = result['density_after'] - result['density_before']
    return result


def main() -> None:
    pd.set_option('display.width', 200)
    pipeline = TrainingPipeline()
    pipeline.run_data_pipeline()
    pipeline.run_labeling_pipeline()
    dataset = pipeline.run_feature_pipeline()
    if dataset is None:
        return

    complication_type = pipeline.target_config.complication_type
    events = pipeline.labels['events']
    events = events[(events['complication_type'] == complication_type) & events['is_positive'] & events['in_drilled_range']]
    step = pipeline.labeler_config.interval_length
    above = pipeline.labeler_config.label_tolerance_above
    below = pipeline.labeler_config.label_tolerance_below

    print(f'\n===== Диагностика меток: {complication_type} =====')
    matched = match_intervals_to_events(dataset, events, step, above, below)
    matched = add_fluid_record_time(matched, pipeline.tables['fluid'])
    matched['features_after_event_start'] = matched['fluid_time'] > matched['event_start'] # утечка: признаки раствора взяты из замера, сделанного уже после начала поглощения (то есть после реакции буровиков)
    print(f'\nПоложительных интервалов: {int(dataset["label"].sum())}, сопоставлено с событиями: {len(matched)}')

    by_offset = matched.groupby('offset').agg(intervals=('label', 'size'), features_after_event_start=('features_after_event_start', 'mean')) # 1. где положительные интервалы относительно глубины события (offset < 0 - выше, 0 - интервал события, > 0 - ниже)
    print('\nПоложение интервала относительно глубины события и доля интервалов, чьи признаки раствора замерены после начала поглощения:')
    print(by_offset.round(2).to_string())
    print(f'Всего положительных интервалов с признаками после начала поглощения: {matched["features_after_event_start"].mean():.0%}')

    columns = [c for c in ['mud_density', 'fit_emw', 'fracture_margin', 'overbalance', 'pore_pressure_emw'] if c in dataset.columns] # 2. признаки положительных интервалов до и после начала поглощения против отрицательных
    wells_with_events = matched['nameWellbore'].unique()
    negatives = dataset[(dataset['label'] == 0) & dataset['nameWellbore'].isin(wells_with_events)]
    comparison = pd.DataFrame({
        'negatives (те же стволы)': negatives[columns].median(),
        'positives: признаки до события': matched.loc[~matched['features_after_event_start'], columns].median(),
        'positives: признаки после начала события': matched.loc[matched['features_after_event_start'], columns].median(),
    })
    print('\nМедианы признаков:')
    print(comparison.round(3).to_string())

    change = mud_density_change(pipeline.tables['fluid'], events) # 3. реакция буровиков: как меняется плотность раствора после поглощения
    if not change.empty:
        print(f'\nПлотность раствора до события и после него (событий с замерами: {len(change)}):')
        print(f"медиана изменения: {change['change'].median():+.3f} sg; "
              f"снизили: {(change['change'] < 0).mean():.0%}, повысили: {(change['change'] > 0).mean():.0%}, "
              f"без изменений: {(change['change'] == 0).mean():.0%}")


if __name__ == '__main__':
    main()