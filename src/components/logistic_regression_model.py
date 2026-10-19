import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.configs.logistic_regression_config import LogisticRegressionConfig
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class LogisticRegressionModel():
    """
    Логистическая регрессия – первая модель, которая учится на интервалах и с осложнениями, и без них,
    то есть оценивает именно P(осложнение | признаки), а не P(признаки | осложнение), как эвристика.
    Логистическая регрессия не умеет работать с пропусками и текстом, поэтому перед ней стоит предобработка:
    1) Числовые признаки – пропуск заменяется медианой train (+ флаг "значение было пропущено", если add_missing_indicator), затем стандартизация;
    2) Категориальные – пропуск становится категорией 'missing', затем one-hot кодирование.
    Предобработка и модель собраны в один sklearn Pipeline: медианы и масштаб считаются только по обучающей части фолда.
    """

    def __init__(self, logistic_regression_config: LogisticRegressionConfig, evaluation_config: EvaluationConfig) -> None:
        self.logistic_regression_config = logistic_regression_config
        self.evaluator = Evaluator(evaluation_config)
        self.model = None
        self.cv_result = None
        self.coefficients = None

    def train(self, dataset: pd.DataFrame, feature_columns: list) -> dict:
        '''Метод прогоняет модель через кросс-валидацию на train, затем обучает итоговую модель на всём train'''
        train = dataset[dataset['split'] == 'train']
        X, y, groups = self._prepare_X(train[feature_columns]), train['label'], train['well_key']

        self.cv_result = self.evaluator.cross_validate(self._fit_predict, X, y, groups)
        self.model = self._build_pipeline(X).fit(X, y)
        self.coefficients = self._extract_coefficients()
        self._print_summary()
        return {'name': 'logistic_regression', 'cv': self.cv_result, 'model': self, 'feature_columns': feature_columns}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''Метод возвращает вероятность осложнения для новых интервалов'''
        if self.model is None:
            raise RuntimeError('Сначала вызовите train()')
        return self.model.predict_proba(self._prepare_X(X))[:, 1]

    def _fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> np.ndarray:
        '''Этот хелпер – функция модели для Evaluator: обучает пайплайн на обучающей части фолда и предсказывает валидацию'''
        return self._build_pipeline(X_train).fit(X_train, y_train).predict_proba(X_val)[:, 1]

    @staticmethod
    def _prepare_X(X: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер приводит категориальные признаки к виду, понятному sklearn: строки с категорией 'missing' вместо пропусков'''
        X = X.copy()
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].astype(object).where(X[column].notna(), 'missing')
        return X

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        '''Этот хелпер собирает предобработку и модель в один sklearn Pipeline'''
        cfg = self.logistic_regression_config
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in X.columns if c not in numeric]

        preprocessing = ColumnTransformer([
            ('numeric', Pipeline([
                ('impute', SimpleImputer(strategy='median', add_indicator=cfg.add_missing_indicator, keep_empty_features=True)),
                ('scale', StandardScaler()),
            ]), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical),
        ])
        model = LogisticRegression(C=cfg.C, class_weight=cfg.class_weight, max_iter=cfg.max_iter)
        return Pipeline([('preprocessing', preprocessing), ('model', model)])

    def _extract_coefficients(self) -> pd.DataFrame:
        '''Этот хелпер достаёт коэффициенты итоговой модели: признаки стандартизованы, поэтому коэффициенты сравнимы между собой'''
        names = self.model.named_steps['preprocessing'].get_feature_names_out()
        coefficients = self.model.named_steps['model'].coef_[0]
        table = pd.DataFrame({'feature': names, 'coefficient': coefficients})
        table['feature'] = table['feature'].str.replace(r'^(numeric|categorical)__', '', regex=True)
        return table.reindex(table['coefficient'].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def _print_summary(self) -> None:
        '''Этот хелпер печатает метрики кросс-валидации и самые весомые признаки'''
        summary = self.cv_result['summary']
        print('\nЛогистическая регрессия, кросс-валидация по скважинам:')
        print(self.cv_result['fold_metrics'].round(3).to_string(index=False))
        print(f"OOF PR-AUC: {summary['oof_pr_auc']:.3f} (случайная модель: {summary['prevalence']:.3f}) | "
              f"PR-AUC по фолдам: медиана {summary['pr_auc_median']:.3f}, {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f} | "
              f"OOF ROC-AUC: {summary['oof_roc_auc']:.3f} | "
              f"OOF Brier: {summary['oof_brier']:.4f} (константа: {summary['prevalence'] * (1 - summary['prevalence']):.4f}) | "
              f"F1 (OOF, порог {summary['threshold']:.3f}): {summary['oof_f1']:.3f}")
        print('\nСамые весомые признаки (коэффициенты на стандартизованной шкале, > 0 – риск выше):')
        print(self.coefficients.head(10).round(3).to_string(index=False))