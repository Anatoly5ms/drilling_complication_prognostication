import numpy as np
import pandas as pd
from src.configs.model_selector_config import ModelSelectorConfig
from src.configs.evaluation_config import EvaluationConfig
from src.utils.evaluator import Evaluator


class RankEnsemble():
    """
    Ансамбль с усреднением рангов. Шкалы вероятностей у моделей разные (лес с весами классов завышает их сильнее бустингов),
    поэтому усредняются не вероятности, а процентили: каждая оценка переводится в долю out-of-fold предсказаний этой модели,
    которые ниже неё. Процентиль 0.97 у любой модели означает одно и то же – «рискованнее 97% интервалов train».
    Шкала для перевода – out-of-fold предсказания на train; по данным здесь ничего не подбирается.
    """

    def __init__(self, members: list[dict]) -> None:
        self.members = members
        self.reference_scores = {m['name']: np.sort(m['cv']['oof_proba']) for m in members} # для каждой модели храним отсортированные OOF-оценки – это её шкала для перевода в процентили
        self.feature_columns = members[0]['feature_columns']

    def to_percentile(self, name: str, scores: np.ndarray) -> np.ndarray:
        '''Метод переводит оценки модели в процентили по её out-of-fold шкале'''
        reference = self.reference_scores[name]
        return np.searchsorted(reference, np.asarray(scores, dtype=float), side='right') / len(reference)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''Метод возвращает оценку риска ансамбля: средний процентиль по моделям-участникам'''
        percentiles = [self.to_percentile(m['name'], m['model'].predict_proba(X[m['feature_columns']])) for m in self.members]
        return np.mean(percentiles, axis=0)


class ModelSelector():
    """
    Отбор моделей и ансамбль:
    1. Порог качества: дальше проходят модели, чей OOF PR-AUC выше эталонной модели (бейзлайна).
    2. Если порог прошли хотя бы min_ensemble_size моделей, из них собирается ансамбль с усреднением рангов
       и оценивается прямо по out-of-fold предсказаниям – на тех же фолдах, что и одиночные модели, без переобучения.
       Ансамбль и есть итоговая модель: модели разной архитектуры сильны в разном, и усреднение объединяет их сильные стороны.
    3. Таблица всех моделей и ансамбля печатается для сравнения: если ансамбль окажется хуже лучшей одиночной модели, это будет видно сразу.
    Тест здесь не используется: он понадобится один раз – для финальной оценки итоговой модели.
    """

    def __init__(self, model_selection_config: ModelSelectorConfig, evaluation_config: EvaluationConfig) -> None:
        self.model_selection_config = model_selection_config
        self.evaluator = Evaluator(evaluation_config)
        self.leaderboard = None
        self.final_model = None

    def select(self, model_results: list[dict], dataset: pd.DataFrame) -> dict:
        '''Метод отбирает модели по порогу и собирает из них ансамбль; таблица всех моделей печатается для сравнения'''
        cfg = self.model_selection_config
        train = dataset[dataset['split'] == 'train']
        y, groups = train['label'], train['well_key']

        reference = next((r for r in model_results if r['name'] == cfg.reference_model), None)
        if reference is None:
            raise ValueError(f'Эталонная модель {cfg.reference_model} не найдена среди обученных')
        threshold = reference['cv']['summary']['oof_pr_auc']
        passed = [r for r in model_results if r['name'] != cfg.reference_model and r['cv']['summary']['oof_pr_auc'] > threshold]

        if len(passed) >= cfg.min_ensemble_size:
            self.final_model = self._build_ensemble(passed, y, groups)
        else:
            self.final_model = passed[0] if passed else reference # ансамбль не из чего собрать, поэтому берём единственную прошедшую порог модель, а если таких нет – эталонную
            print(f'Порог прошли моделей: {len(passed)} < {cfg.min_ensemble_size}, ансамбль не собирается')

        ensembles = [self.final_model] if self.final_model['name'].startswith('ensemble') else []
        self.leaderboard = self._build_leaderboard(model_results, ensembles, threshold)
        self._print_summary(threshold)
        return self.final_model

    def _build_ensemble(self, members: list[dict], y: pd.Series, groups: pd.Series) -> dict:
        '''Этот хелпер собирает ансамбль и оценивает его по усреднённым OOF-процентилям моделей-участников'''
        ensemble = RankEnsemble(members)
        oof_percentiles = [ensemble.to_percentile(m['name'], m['cv']['oof_proba']) for m in members]
        cv = self.evaluator.evaluate_oof(np.mean(oof_percentiles, axis=0), y, groups)
        name = 'ensemble(' + ' + '.join(m['name'] for m in members) + ')'
        return {'name': name, 'cv': cv, 'model': ensemble, 'feature_columns': ensemble.feature_columns}

    def _build_leaderboard(self, model_results: list[dict], ensembles: list[dict], threshold: float) -> pd.DataFrame:
        '''Этот хелпер собирает таблицу всех моделей и ансамбля с основными метриками'''
        rows = []
        for result in model_results + ensembles:
            summary = result['cv']['summary']
            rows.append({
                'model': result['name'],
                'oof_pr_auc': summary['oof_pr_auc'],
                'pr_auc_median': summary['pr_auc_median'],
                'oof_roc_auc': summary['oof_roc_auc'],
                'oof_f1': summary['oof_f1'],
                'passed': result['name'] != self.model_selection_config.reference_model and summary['oof_pr_auc'] > threshold,
            })
        return pd.DataFrame(rows).sort_values('oof_pr_auc', ascending=False).reset_index(drop=True)

    def _print_summary(self, threshold: float) -> None:
        '''Этот хелпер печатает таблицу моделей и итоговую модель'''
        print(f'\nОтбор моделей (порог – OOF PR-AUC {self.model_selection_config.reference_model}: {threshold:.3f}):')
        print(self.leaderboard.round(3).to_string(index=False))
        print(f"\nИтоговая модель: {self.final_model['name']} (OOF PR-AUC {self.final_model['cv']['summary']['oof_pr_auc']:.3f})")
        if self.final_model['name'].startswith('ensemble'):
            print('\nАнсамбль по фолдам:')
            print(self.final_model['cv']['fold_metrics'].round(3).to_string(index=False))