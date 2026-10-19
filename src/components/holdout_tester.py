import numpy as np
import pandas as pd
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class HoldoutTester():
    """
    Однократная оценка на отложенных скважинах, которые ни разу не участвовали ни в обучении, ни в выборе моделей.
    Для каждой модели используется её порог из кросс-валидации: на тесте ничего не подбирается.
    Итоговая модель – ансамбль, остальные модели оцениваются для контекста (с ними ничего не выбирается и не меняется).
    Метрики даются с доверительными интервалами бутстрэпом, плюс разбивка итоговой модели по скважинам.
    """

    def __init__(self, evaluation_config: EvaluationConfig) -> None:
        self.evaluator = Evaluator(evaluation_config)
        self.evaluation_config = evaluation_config
        self.results = None
        self.by_well = None

    def test(self, model_results: list[dict], final_model: dict, dataset: pd.DataFrame) -> pd.DataFrame:
        '''Метод оценивает все модели и итоговую модель на отложенных скважинах и печатает результаты'''
        test = dataset[dataset['split'] == 'test']
        y, groups = test['label'], test['well_key']

        candidates = model_results if any(r is final_model for r in model_results) else model_results + [final_model] # если ансамбль не собирался, то итоговая модель – это одна из обученных, и она уже есть в списке
        rows, final_proba = [], None
        for result in candidates:
            proba = result['model'].predict_proba(test[result['feature_columns']])
            metrics = self.evaluator.evaluate_test(y, proba, groups, result['cv']['threshold'])
            rows.append(self._to_row(result['name'], metrics, result is final_model))
            if result is final_model:
                final_proba = proba

        self.results = pd.DataFrame(rows).sort_values('pr_auc', ascending=False).reset_index(drop=True)
        self.by_well = self._evaluate_by_well(test, final_proba)
        self._print_summary(test)
        return self.results

    @staticmethod
    def _to_row(name: str, metrics: pd.DataFrame, is_final: bool) -> dict:
        '''Этот хелпер собирает строку итоговой таблицы: точечные значения и доверительные интервалы'''
        row = {'model': name + (' [итоговая]' if is_final else '')}
        for metric in ['pr_auc', 'roc_auc']:
            row[metric] = metrics.loc[metric, 'value']
            row[f'{metric}_ci'] = f"[{metrics.loc[metric, 'ci_low']:.3f}; {metrics.loc[metric, 'ci_high']:.3f}]"
        for metric in ['f1', 'precision', 'recall']:
            row[metric] = metrics.loc[metric, 'value']
        return row

    def _evaluate_by_well(self, test: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
        '''Этот хелпер считает качество итоговой модели отдельно по каждой тестовой скважине'''
        frame = test[['well_key', 'label']].assign(proba=proba)
        rows = []
        for well, part in frame.groupby('well_key'):
            metrics = self.evaluator._threshold_free_metrics(part['label'].to_numpy(), part['proba'].to_numpy())
            rows.append({'well': well, 'intervals': len(part), 'positives': int(part['label'].sum()),
                         'prevalence': part['label'].mean(), 'pr_auc': metrics['pr_auc'], 'roc_auc': metrics['roc_auc']})
        return pd.DataFrame(rows)

    def _print_summary(self, test: pd.DataFrame) -> None:
        '''Этот хелпер печатает результаты на тесте'''
        cfg = self.evaluation_config
        print(f"\n===== Отложенный тест: {test['well_key'].nunique()} скважины, {len(test)} интервалов, "
              f"{int(test['label'].sum())} положительных (случайная модель: PR-AUC {test['label'].mean():.3f}) =====")
        print(f'Доверительные интервалы {cfg.confidence_level:.0%}, бутстрэп по {"скважинам" if cfg.bootstrap_unit == "group" else "интервалам"}; '
              f'пороговые метрики – при пороге из кросс-валидации')
        print(self.results.round(3).to_string(index=False))
        print('\nИтоговая модель по скважинам:')
        print(self.by_well.round(3).to_string(index=False))