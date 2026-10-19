import numpy as np
import pandas as pd
from src.configs.baseline_config import BaselineConfig
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class BaselineCalculator():
    """
    Бейзлайн – вероятностная эвристика из магистерской диссертации (2019), перенесённая на данные Volve:
    1. Каждый признак делится на группы (числовые – по квантилям train, категориальные – по значениям, пропуск – отдельная группа).
    2. Для каждой группы считается P(группа | осложнение): доля интервалов с осложнением, попавших в эту группу;
    3. Риск интервала = произведение этих вероятностей по всем признакам (теорема умножения вероятностей).
    Это намеренно точная копия исходного метода, включая его ограничения: признаки считаются независимыми,
    а отрицательные примеры не участвуют. Поэтому оценка ранжирует интервалы, но не является вероятностью:
    PR-AUC и ROC-AUC корректны, Brier score для бейзлайна не интерпретируется.
    Все статистики считаются только на обучающей части фолда, иначе сравнение с моделями было бы нечестным.
    """

    MISSING = 'missing'
    UNSEEN = 'unseen'

    def __init__(self, baseline_config: BaselineConfig, evaluation_config: EvaluationConfig) -> None:
        self.baseline_config = baseline_config
        self.evaluator = Evaluator(evaluation_config)
        self.model = None
        self.cv_result = None

    def train(self, dataset: pd.DataFrame, feature_columns: list) -> dict:
        '''Метод прогоняет эвристику через кросс-валидацию на train, затем строит итоговые таблицы вероятностей на всём train'''
        train = dataset[dataset['split'] == 'train']
        X, y, groups = train[feature_columns], train['label'], train['well_key']

        self.cv_result = self.evaluator.cross_validate(self._fit_predict, X, y, groups)
        self.model = self._fit(X, y)
        self._print_summary()
        return {'name': 'baseline', 'cv': self.cv_result, 'model': self, 'feature_columns': feature_columns}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''Метод возвращает оценку риска для новых интервалов по таблицам, построенным на всём train'''
        if self.model is None:
            raise RuntimeError('Сначала вызовите train()')
        return self._score(X, self.model)

    def _fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> np.ndarray:
        '''Этот хелпер – функция модели для Evaluator: строит таблицы на обучающей части фолда и оценивает валидацию'''
        return self._score(X_val, self._fit(X_train, y_train))

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        '''Этот хелпер строит для каждого признака границы групп и логарифмы P(группа | осложнение)'''
        cfg = self.baseline_config
        positives = y.to_numpy() == 1
        n_positive = positives.sum()
        model = {}

        for column in X.columns:
            series = X[column]
            edges = self._bin_edges(series) if pd.api.types.is_numeric_dtype(series) else None
            bins = self._assign_bins(series, edges)
            categories = sorted(set(bins.unique()) | {self.MISSING, self.UNSEEN})
            counts = bins[positives].value_counts().reindex(categories, fill_value=0)
            log_prob = np.log((counts + cfg.smoothing) / (n_positive + cfg.smoothing * len(categories))) # P(группа | осложнение) со сглаживанием Лапласа. Храним логарифм, чтобы произведение не ушло в машинный ноль
            model[column] = {'edges': edges, 'log_prob': log_prob}
        return model

    def _score(self, X: pd.DataFrame, model: dict) -> np.ndarray:
        '''Этот хелпер считает риск интервала: произведение P(группа | осложнение) по всем признакам (через сумму логарифмов)'''
        log_score = np.zeros(len(X))
        for column, params in model.items():
            bins = self._assign_bins(X[column], params['edges'])
            bins = bins.where(bins.isin(params['log_prob'].index), self.UNSEEN)  # группа, которой не было в train
            log_score += params['log_prob'].reindex(bins).to_numpy()
        return np.exp(log_score)

    def _bin_edges(self, series: pd.Series) -> np.ndarray:
        '''Этот хелпер считает внутренние границы квантильных групп числового признака по train'''
        values = series.dropna()
        if values.empty:
            return np.array([])
        quantiles = np.linspace(0, 1, self.baseline_config.n_bins + 1)[1:-1]
        return np.unique(np.quantile(values, quantiles))  # одинаковые квантили (много повторов) схлопываются

    def _assign_bins(self, series: pd.Series, edges: np.ndarray | None) -> pd.Series:
        '''Этот хелпер относит значения к группам: числовые – по границам, категориальные – по значению, пропуск – в missing'''
        if edges is None:
            bins = series.astype('string')
        else:
            bins = pd.Series(np.digitize(series.to_numpy(dtype=float), edges), index=series.index).astype('string')
        return bins.where(series.notna(), self.MISSING).astype(object)

    def _print_summary(self) -> None:
        '''Этот хелпер печатает метрики кросс-валидации'''
        summary = self.cv_result['summary']
        print('\nБейзлайн (эвристика диссертации), кросс-валидация по скважинам:')
        print(self.cv_result['fold_metrics'].round(3).to_string(index=False))
        print(f"OOF PR-AUC: {summary['oof_pr_auc']:.3f} (случайная модель: {summary['prevalence']:.3f}) | "
              f"PR-AUC по фолдам: медиана {summary['pr_auc_median']:.3f}, {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f} | "
              f"OOF ROC-AUC: {summary['oof_roc_auc']:.3f} | "
              f"F1 (OOF, порог {summary['threshold']:.2e}): {summary['oof_f1']:.3f}")