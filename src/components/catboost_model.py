import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from src.configs.catboost_config import CatBoostConfig
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class CatBoostModel():
    """
    Градиентный бустинг CatBoost. Два отличия от LightGBM, важные для наших данных:
    1) Симметричные деревья: на каждом уровне дерева используется одно и то же условие для всех веток. Деревья получаются проще и устойчивее – это встроенная защита от переобучения на малых выборках.
    2) Категориальные признаки кодируются статистикой целевой переменной по категории, посчитанной так, чтобы интервал не «видел» собственную метку (упорядоченное кодирование) – без утечки внутри обучения.
    Предобработка:
    1) Числовые признаки – как есть, пропуски CatBoost обрабатывает сам.
    2) Категориальные – пропуск становится категорией 'missing' (CatBoost не принимает пропуски в категориях).
    Новые категории в валидации CatBoost обрабатывает сам.
    """

    def __init__(self, catboost_config: CatBoostConfig, evaluation_config: EvaluationConfig) -> None:
        self.catboost_config = catboost_config
        self.evaluator = Evaluator(evaluation_config)
        self.model = None
        self.cv_result = None
        self.importances = None

    def train(self, dataset: pd.DataFrame, feature_columns: list) -> dict:
        '''Метод прогоняет модель через кросс-валидацию на train, затем обучает итоговую модель на всём train'''
        train = dataset[dataset['split'] == 'train']
        X, y, groups = self._prepare_X(train[feature_columns]), train['label'], train['well_key']

        self.cv_result = self.evaluator.cross_validate(self._fit_predict, X, y, groups)
        self.model = self._fit(X, y)
        self.importances = self._extract_importances(feature_columns)
        self._print_summary()
        return {'name': 'catboost', 'cv': self.cv_result, 'model': self, 'feature_columns': feature_columns}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''Метод возвращает оценку риска осложнения для новых интервалов'''
        if self.model is None:
            raise RuntimeError('Сначала вызовите train()')
        return self.model.predict_proba(self._prepare_X(X))[:, 1]

    def _fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> np.ndarray:
        '''Этот хелпер – функция модели для Evaluator: обучает модель на обучающей части фолда и предсказывает валидацию'''
        return self._fit(X_train, y_train).predict_proba(X_val)[:, 1]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
        '''Этот хелпер обучает модель, передавая ей список категориальных признаков'''
        cfg = self.catboost_config
        categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        model = CatBoostClassifier(
            iterations=cfg.iterations, learning_rate=cfg.learning_rate, depth=cfg.depth, l2_leaf_reg=cfg.l2_leaf_reg,
            auto_class_weights=cfg.auto_class_weights, random_seed=cfg.random_seed,
            verbose=False, allow_writing_files=False,  # без логов обучения и папки catboost_info в корне проекта
        )
        return model.fit(X, y, cat_features=categorical)

    @staticmethod
    def _prepare_X(X: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер приводит категориальные признаки к строкам с категорией 'missing' вместо пропусков'''
        X = X.copy()
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].astype(object).where(X[column].notna(), 'missing').astype(str)
        return X

    def _extract_importances(self, feature_columns: list) -> pd.DataFrame:
        '''Этот хелпер достаёт важность признаков итоговой модели (насколько в среднем меняется прогноз при изменении признака)'''
        importance = self.model.get_feature_importance()
        table = pd.DataFrame({'feature': feature_columns, 'importance': importance / importance.sum()})
        return table.sort_values('importance', ascending=False).reset_index(drop=True)

    def _print_summary(self) -> None:
        '''Этот хелпер печатает метрики кросс-валидации и самые важные признаки'''
        summary = self.cv_result['summary']
        print('\nCatBoost, кросс-валидация по скважинам:')
        print(self.cv_result['fold_metrics'].round(3).to_string(index=False))
        print(f"OOF PR-AUC: {summary['oof_pr_auc']:.3f} (случайная модель: {summary['prevalence']:.3f}) | "
              f"PR-AUC по фолдам: медиана {summary['pr_auc_median']:.3f}, {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f} | "
              f"OOF ROC-AUC: {summary['oof_roc_auc']:.3f} | "
              f"F1 (OOF, порог {summary['threshold']:.3f}): {summary['oof_f1']:.3f}")
        print('\nСамые важные признаки (доля влияния на прогноз):')
        print(self.importances.head(10).round(3).to_string(index=False))