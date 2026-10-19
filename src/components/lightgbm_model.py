import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from src.configs.lightgbm_config import LightGBMConfig
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class LightGBMModel():
    """
    Градиентный бустинг LightGBM: деревья строятся последовательно, и каждое следующее исправляет ошибки предыдущих
    (лес, наоборот, строит деревья независимо и усредняет). Обычно это сильнее леса на табличных данных,
    но и легче переобучается на малых выборках – поэтому параметры в конфиге консервативные.
    Предобработка минимальная, LightGBM умеет всё сам:
    1) Числовые признаки – как есть, пропуски обрабатываются нативно (для них выбирается лучшая ветка разбиения);
    2) Категориальные – тип category с набором категорий из обучающей части; LightGBM разбивает их по группам категорий, а не по порядковым номерам. Категория, которой не было в обучении, становится пропуском.
    Ранняя остановка не используется: для неё нужна отдельная валидация, а валидационный фолд занят оценкой модели.
    """

    def __init__(self, lightgbm_config: LightGBMConfig, evaluation_config: EvaluationConfig) -> None:
        self.lightgbm_config = lightgbm_config
        self.evaluator = Evaluator(evaluation_config)
        self.model = None
        self.categories = None
        self.cv_result = None
        self.importances = None

    def train(self, dataset: pd.DataFrame, feature_columns: list) -> dict:
        '''Метод прогоняет модель через кросс-валидацию на train, затем обучает итоговую модель на всём train'''
        train = dataset[dataset['split'] == 'train']
        X, y, groups = train[feature_columns], train['label'], train['well_key']

        self.cv_result = self.evaluator.cross_validate(self._fit_predict, X, y, groups)
        self.model, self.categories = self._fit(X, y)
        self.importances = self._extract_importances(feature_columns)
        self._print_summary()
        return {'name': 'lightgbm', 'cv': self.cv_result, 'model': self, 'feature_columns': feature_columns}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''Метод возвращает оценку риска осложнения для новых интервалов'''
        if self.model is None:
            raise RuntimeError('Сначала вызовите train()')
        return self.model.predict_proba(self._prepare_X(X, self.categories))[:, 1]

    def _fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> np.ndarray:
        '''Этот хелпер – функция модели для Evaluator: обучает модель на обучающей части фолда и предсказывает валидацию'''
        model, categories = self._fit(X_train, y_train)
        return model.predict_proba(self._prepare_X(X_val, categories))[:, 1]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> tuple[LGBMClassifier, dict]:
        '''Этот хелпер запоминает категории из обучающей выборки и обучает модель'''
        categories = {c: sorted(X[c].dropna().astype(str).unique()) for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])}
        cfg = self.lightgbm_config
        model = LGBMClassifier(
            n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples, subsample=cfg.subsample, subsample_freq=cfg.subsample_freq,
            colsample_bytree=cfg.colsample_bytree, reg_lambda=cfg.reg_lambda, class_weight=cfg.class_weight,
            random_state=cfg.random_state, importance_type='gain', verbose=-1,
        )
        return model.fit(self._prepare_X(X, categories), y), categories

    @staticmethod
    def _prepare_X(X: pd.DataFrame, categories: dict) -> pd.DataFrame:
        '''Этот хелпер переводит категориальные признаки в тип category с фиксированным набором категорий из обучения'''
        X = X.copy()
        for column, values in categories.items():
            X[column] = pd.Categorical(X[column].astype(object).where(X[column].notna(), None), categories=values)
        return X

    def _extract_importances(self, feature_columns: list) -> pd.DataFrame:
        '''Этот хелпер достаёт важность признаков итоговой модели: суммарный прирост качества (gain) от разбиений по признаку'''
        gain = self.model.feature_importances_
        table = pd.DataFrame({'feature': feature_columns, 'importance': gain / gain.sum() if gain.sum() else gain})
        return table.sort_values('importance', ascending=False).reset_index(drop=True)

    def _print_summary(self) -> None:
        '''Этот хелпер печатает метрики кросс-валидации и самые важные признаки'''
        summary = self.cv_result['summary']
        print('\nLightGBM, кросс-валидация по скважинам:')
        print(self.cv_result['fold_metrics'].round(3).to_string(index=False))
        print(f"OOF PR-AUC: {summary['oof_pr_auc']:.3f} (случайная модель: {summary['prevalence']:.3f}) | "
              f"PR-AUC по фолдам: медиана {summary['pr_auc_median']:.3f}, {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f} | "
              f"OOF ROC-AUC: {summary['oof_roc_auc']:.3f} | "
              f"F1 (OOF, порог {summary['threshold']:.3f}): {summary['oof_f1']:.3f}")
        print('\nСамые важные признаки (доля в суммарном приросте качества, gain):')
        print(self.importances.head(10).round(3).to_string(index=False))