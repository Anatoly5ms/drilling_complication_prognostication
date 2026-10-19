import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from src.configs.random_forest_config import RandomForestConfig
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class RandomForestModel():
    """
    Случайный лес – первая модель, которая учитывает взаимодействия признаков: каждое дерево делит интервалы по порогам
    ("репрессия > 0,4 и открытый ствол > 800 м"), а лес усредняет сотни таких деревьев, обученных на разных подвыборках.
    По сути, это строгая версия идеи эвристики 2019 года: группы по порогам, но пороги и их сочетания подбираются по данным.
    Предобработка минимальная:
    1) Числовые признаки как есть: леса в sklearn (>= 1.4) сами работают с пропусками, выбирая для них лучшую ветку разбиения.
    2) Категориальные признаки. Пропуск становится категорией 'missing', затем порядковое кодирование (деревьям достаточно номера категории).
    Стандартизация не нужна: деревьям важен только порядок значений, а не их масштаб.
    """

    def __init__(self, random_forest_config: RandomForestConfig, evaluation_config: EvaluationConfig) -> None:
        self.random_forest_config = random_forest_config
        self.evaluator = Evaluator(evaluation_config)
        self.model = None
        self.cv_result = None
        self.importances = None

    def train(self, dataset: pd.DataFrame, feature_columns: list) -> dict:
        '''Метод прогоняет модель через кросс-валидацию на train, затем обучает итоговую модель на всём train'''
        train = dataset[dataset['split'] == 'train']
        X, y, groups = self._prepare_X(train[feature_columns]), train['label'], train['well_key']

        self.cv_result = self.evaluator.cross_validate(self._fit_predict, X, y, groups)
        self.model = self._build_pipeline(X).fit(X, y)
        self.importances = self._extract_importances()
        self._print_summary()
        return {'name': 'random_forest', 'cv': self.cv_result, 'model': self, 'feature_columns': feature_columns}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''Метод возвращает оценку риска осложнения для новых интервалов'''
        if self.model is None:
            raise RuntimeError('Сначала вызовите train()')
        return self.model.predict_proba(self._prepare_X(X))[:, 1]

    def _fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> np.ndarray:
        '''Этот хелпер – функция модели для Evaluator. Она бучает пайплайн на обучающей части фолда и предсказывает валидацию'''
        return self._build_pipeline(X_train).fit(X_train, y_train).predict_proba(X_val)[:, 1]

    @staticmethod
    def _prepare_X(X: pd.DataFrame) -> pd.DataFrame:
        '''Этот хелпер приводит категориальные признаки к строкам с категорией 'missing' вместо пропусков'''
        X = X.copy()
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].astype(object).where(X[column].notna(), 'missing')
        return X

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        '''Этот хелпер собирает предобработку и модель в один sklearn Pipeline'''
        cfg = self.random_forest_config
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in X.columns if c not in numeric]

        preprocessing = ColumnTransformer([
            ('numeric', 'passthrough', numeric),
            ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical), # категория, которой не было в обучающей части, получает код -1
        ])
        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators, min_samples_leaf=cfg.min_samples_leaf, max_depth=cfg.max_depth,
            max_features=cfg.max_features, class_weight=cfg.class_weight, random_state=cfg.random_state, n_jobs=-1,
        )
        return Pipeline([('preprocessing', preprocessing), ('model', model)])

    def _extract_importances(self) -> pd.DataFrame:
        '''Этот хелпер достаёт важность признаков итоговой модели (среднее уменьшение неоднородности в разбиениях).
        Оговорка: такая важность завышает числовые признаки с большим числом уникальных значений'''
        names = self.model.named_steps['preprocessing'].get_feature_names_out()
        table = pd.DataFrame({'feature': names, 'importance': self.model.named_steps['model'].feature_importances_})
        table['feature'] = table['feature'].str.replace(r'^(numeric|categorical)__', '', regex=True)
        return table.sort_values('importance', ascending=False).reset_index(drop=True)

    def _print_summary(self) -> None:
        '''Этот хелпер печатает метрики кросс-валидации и самые важные признаки'''
        summary = self.cv_result['summary']
        print('\nСлучайный лес, кросс-валидация по скважинам:')
        print(self.cv_result['fold_metrics'].round(3).to_string(index=False))
        print(f"OOF PR-AUC: {summary['oof_pr_auc']:.3f} (случайная модель: {summary['prevalence']:.3f}) | "
              f"PR-AUC по фолдам: медиана {summary['pr_auc_median']:.3f}, {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f} | "
              f"OOF ROC-AUC: {summary['oof_roc_auc']:.3f} | "
              f"F1 (OOF, порог {summary['threshold']:.3f}): {summary['oof_f1']:.3f}")
        print('\nСамые важные признаки (доля в уменьшении неоднородности):')
        print(self.importances.head(10).round(3).to_string(index=False))