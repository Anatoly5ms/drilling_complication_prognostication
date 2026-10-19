from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from src.configs.evaluation_config import EvaluationConfig


FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray] # функция модели для кросс-валидации: (X_train, y_train, X_val) -> вероятности положительного класса для X_val


class Evaluator():
    """
    Единая оценка для всех моделей: одинаковые фолды, одинаковые метрики, одинаковый подбор порога.
    cross_validate – кросс-валидация по скважинам внутри train: out-of-fold вероятности, метрики по фолдам, порог.
    evaluate_test – метрики на отложенном тесте при пороге из CV + доверительные интервалы бутстрэпом.
    Evaluator ничего не знает о моделях: модель передаётся функцией fit_predict, которая обучается на фолде
    и возвращает вероятности для валидации. Так любая модель (от эвристики до CatBoost) оценивается одним кодом.
    """

    THRESHOLD_FREE_METRICS = ['pr_auc', 'roc_auc', 'brier']
    THRESHOLD_METRICS = ['f1', 'precision', 'recall']

    def __init__(self, evaluation_config: EvaluationConfig) -> None:
        self.evaluation_config = evaluation_config

    def get_folds(self, y: pd.Series, groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        '''Метод строит фолды StratifiedGroupKFold: скважина целиком попадает в один фолд, и в каждом фолде есть положительные примеры.
        Если при заданном числе фолдов так разбить не удаётся, перебираются другие random_state, а затем число фолдов уменьшается'''
        cfg = self.evaluation_config
        positive_groups = groups[y == 1].nunique()
        if positive_groups < 2:
            raise ValueError(f'Для кросс-валидации нужно минимум 2 скважины с осложнениями в train, найдено: {positive_groups}')

        for n_splits in range(min(cfg.n_splits, positive_groups), 1, -1):
            for attempt in range(50):
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state + attempt)
                folds = list(splitter.split(np.zeros(len(y)), y, groups))
                if all(y.iloc[val_idx].sum() > 0 for _, val_idx in folds):
                    if n_splits < cfg.n_splits:
                        print(f'Фолдов {n_splits} вместо {cfg.n_splits}: так в каждом фолде есть положительные примеры')
                    return folds
        raise ValueError('Не удалось построить фолды с положительными примерами в каждом')

    def cross_validate(self, fit_predict: FitPredict, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
        '''Метод прогоняет модель по фолдам и возвращает out-of-fold вероятности, метрики по фолдам, сводку и порог.
        Для сравнения моделей используется oof_pr_auc – PR-AUC по объединённым out-of-fold предсказаниям'''
        X, y, groups = X.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
        folds = self.get_folds(y, groups)
        oof_proba = np.full(len(y), np.nan)
        for train_idx, val_idx in folds:
            oof_proba[val_idx] = np.asarray(fit_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx]), dtype=float)
        return self.evaluate_oof(oof_proba, y, groups, folds)

    def evaluate_oof(self, oof_proba: np.ndarray, y: pd.Series, groups: pd.Series, folds: list | None = None) -> dict:
        '''Метод оценивает готовые out-of-fold предсказания: метрики по фолдам, сводку и порог.
        Нужен, чтобы оценить ансамбль по OOF-предсказаниям моделей без повторного обучения'''
        y, groups = y.reset_index(drop=True), groups.reset_index(drop=True)
        oof_proba = np.asarray(oof_proba, dtype=float)
        folds = folds if folds is not None else self.get_folds(y, groups)

        fold_rows = []
        for fold, (_, val_idx) in enumerate(folds):
            row = {'fold': fold, 'val_rows': len(val_idx), 'val_positives': int(y.iloc[val_idx].sum()),
                   'val_wells': groups.iloc[val_idx].nunique()}
            row.update(self._threshold_free_metrics(y.iloc[val_idx].to_numpy(), oof_proba[val_idx]))
            fold_rows.append(row)

        fold_metrics = pd.DataFrame(fold_rows)
        threshold = self.tune_threshold(y.to_numpy(), oof_proba)
        oof_metrics = self.compute_metrics(y.to_numpy(), oof_proba, threshold)
        summary = {
            **{f'{m}_mean': float(fold_metrics[m].mean()) for m in self.THRESHOLD_FREE_METRICS},
            **{f'{m}_std': float(fold_metrics[m].std()) for m in self.THRESHOLD_FREE_METRICS},
            **{f'{m}_median': float(fold_metrics[m].median()) for m in self.THRESHOLD_FREE_METRICS},
            **{f'oof_{m}': float(oof_metrics[m]) for m in self.THRESHOLD_FREE_METRICS}, # метрики по объединённым out-of-fold предсказаниям всего train: один удачный маленький фолд их не перекашивает
            **{f'oof_{m}': float(oof_metrics[m]) for m in self.THRESHOLD_METRICS},
            'threshold': threshold,
            'prevalence': float(y.mean()),  # базовый уровень PR-AUC: столько даёт случайная модель
        }
        return {'oof_proba': oof_proba, 'fold_metrics': fold_metrics, 'summary': summary, 'threshold': threshold}

    def tune_threshold(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        '''Метод подбирает порог, при котором F1 максимален (по out-of-fold предсказаниям train)'''
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best = np.argmax(f1[:-1]) if len(thresholds) else 0  # последняя точка кривой не имеет порога
        return float(thresholds[best]) if len(thresholds) else 0.5

    def compute_metrics(self, y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
        '''Метод считает все метрики: беспороговые (PR-AUC, ROC-AUC, Brier) и пороговые (F1, precision, recall)'''
        predicted = (proba >= threshold).astype(int)
        metrics = self._threshold_free_metrics(y_true, proba)
        metrics.update({
            'f1': f1_score(y_true, predicted, zero_division=0),
            'precision': precision_score(y_true, predicted, zero_division=0),
            'recall': recall_score(y_true, predicted, zero_division=0),
        })
        return metrics

    def evaluate_test(self, y_true: pd.Series, proba: np.ndarray, groups: pd.Series, threshold: float) -> pd.DataFrame:
        '''Метод считает метрики на отложенном тесте при пороге из CV и доверительные интервалы бутстрэпом'''
        cfg = self.evaluation_config
        y_true, groups, proba = y_true.to_numpy(), groups.to_numpy(), np.asarray(proba, dtype=float)
        point = self.compute_metrics(y_true, proba, threshold)

        rng = np.random.default_rng(cfg.random_state)
        unique_groups = np.unique(groups)
        samples = []
        for _ in range(cfg.bootstrap_iterations):
            if cfg.bootstrap_unit == 'group':
                chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                idx = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
            else:
                idx = rng.integers(0, len(y_true), size=len(y_true))
            if y_true[idx].sum() == 0:
                continue  # без положительных примеров PR-AUC не определён
            samples.append(self.compute_metrics(y_true[idx], proba[idx], threshold))

        samples = pd.DataFrame(samples)
        alpha = (1 - cfg.confidence_level) / 2
        result = pd.DataFrame({
            'value': pd.Series(point),
            'ci_low': samples.quantile(alpha),
            'ci_high': samples.quantile(1 - alpha),
        })
        result.loc['prevalence'] = [y_true.mean(), np.nan, np.nan]
        return result

    @staticmethod
    def _threshold_free_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
        '''Этот хелпер считает метрики, не зависящие от порога; если в выборке один класс, PR-AUC и ROC-AUC не определены'''
        both_classes = len(np.unique(y_true)) == 2
        return {
            'pr_auc': average_precision_score(y_true, proba) if both_classes else np.nan,
            'roc_auc': roc_auc_score(y_true, proba) if both_classes else np.nan,
            'brier': brier_score_loss(y_true, proba),
        }