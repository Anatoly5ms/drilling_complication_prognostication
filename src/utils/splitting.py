import numpy as np
import pandas as pd


def assign_holdout_split(df: pd.DataFrame, group_column: str, label_column: str, test_size: float, random_state: int, n_trials: int = 2000) -> pd.Series:
    """
    Делит данные на train/test по группам (скважинам), чтобы интервалы одной скважины не попали в обе части.
    Скважины сильно различаются по длине, поэтому набор тестовых скважин подбирается перебором случайных комбинаций:
    выбирается тот, где и доля интервалов, и доля положительных примеров в тесте ближе всего к test_size.
    В обеих частях обязательно есть положительные примеры. При одном random_state результат всегда одинаковый.
    Возвращает колонку split со значениями 'train' / 'test'.
    """
    rng = np.random.default_rng(random_state)
    groups = df.groupby(group_column)[label_column].agg(['size', 'sum'])
    total_rows, total_positives = groups['size'].sum(), groups['sum'].sum()
    max_test_groups = max(1, len(groups) // 2)  # в тест не больше половины скважин

    best_groups, best_score = None, np.inf
    for _ in range(n_trials):
        n_test = rng.integers(1, max_test_groups + 1)
        chosen = rng.choice(groups.index.to_numpy(), size=n_test, replace=False)
        test_positives = groups.loc[chosen, 'sum'].sum()
        if test_positives == 0 or test_positives == total_positives:
            continue  # положительные примеры нужны и в train, и в test
        rows_share = groups.loc[chosen, 'size'].sum() / total_rows
        positives_share = test_positives / total_positives
        score = abs(rows_share - test_size) + abs(positives_share - test_size)
        if score < best_score:
            best_groups, best_score = chosen, score

    if best_groups is None:
        raise ValueError('Не удалось подобрать тестовые скважины с положительными примерами в обеих частях')
    return pd.Series(np.where(df[group_column].isin(best_groups), 'test', 'train'), index=df.index, name='split')