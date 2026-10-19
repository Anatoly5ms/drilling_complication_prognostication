import pandas as pd

from src.components.data_builder import DataBuilder
from src.configs.data_config import get_data_config

from src.components.labeler import Labeler
from src.configs.labeler_config import get_labeler_config
from src.configs.target_config import get_target_config

from src.components.feature_builder import FeatureBuilder
from src.configs.feature_config import get_feature_config

from src.components.baseline_calculator import BaselineCalculator
from src.configs.baseline_config import get_baseline_config
from src.configs.evaluation_config import get_evaluation_config

from src.components.logistic_regression_model import LogisticRegressionModel
from src.configs.logistic_regression_config import get_logistic_regression_config

from src.components.random_forest_model import RandomForestModel
from src.configs.random_forest_config import get_random_forest_config

from src.components.lightgbm_model import LightGBMModel
from src.configs.lightgbm_config import get_lightgbm_config

from src.components.catboost_model import CatBoostModel
from src.configs.catboost_config import get_catboost_config

from src.components.model_selector import ModelSelector
from src.configs.model_selector_config import get_model_selector_config

from src.components.holdout_tester import HoldoutTester


class TrainingPipeline():
    """
    Обучающий пайплайн: DataBuilder -> Labeler -> FeatureBuilder -> модели (бейзлайн, логистическая регрессия, остальные по мере готовности).
    Если событий выбранного типа осложнения меньше порога, пайплайн останавливается после разметки.
    """

    def __init__(self) -> None:
        self.data_config = get_data_config()
        self.labeler_config = get_labeler_config()
        self.target_config = get_target_config()
        self.feature_config = get_feature_config()
        self.evaluation_config = get_evaluation_config()
        self.baseline_config = get_baseline_config()
        self.logistic_regression_config = get_logistic_regression_config()
        self.random_forest_config = get_random_forest_config()
        self.lightgbm_config = get_lightgbm_config()
        self.catboost_config = get_catboost_config()
        self.model_selector_config = get_model_selector_config()

        self.tables = None
        self.labels = None
        self.has_enough_events = False
        self.dataset = None
        self.feature_columns = None
        self.model_results = []
        self.final_model = None

    def run_data_pipeline(self) -> dict[str, pd.DataFrame]:
        self.tables = DataBuilder(self.data_config).build_tables()
        return self.tables

    def run_labeling_pipeline(self) -> dict[str, pd.DataFrame]:
        labeler = Labeler(self.labeler_config, self.target_config, self.tables)
        self.labels = labeler.build_labels()
        self.has_enough_events = labeler.has_enough_events()
        return self.labels

    def run_feature_pipeline(self) -> pd.DataFrame | None:
        if not self.has_enough_events:
            return None
        feature_builder = FeatureBuilder(self.feature_config, self.target_config, self.tables, self.labels['intervals'])
        self.dataset = feature_builder.build_features()
        self.feature_columns = feature_builder.feature_columns
        return self.dataset

    def run_model_pipeline(self) -> list[dict] | None:
        if self.dataset is None:
            return None
        baseline = BaselineCalculator(self.baseline_config, self.evaluation_config)
        self.model_results.append(baseline.train(self.dataset, self.feature_columns))

        logistic_regression = LogisticRegressionModel(self.logistic_regression_config, self.evaluation_config)
        self.model_results.append(logistic_regression.train(self.dataset, self.feature_columns))

        random_forest = RandomForestModel(self.random_forest_config, self.evaluation_config)
        self.model_results.append(random_forest.train(self.dataset, self.feature_columns))

        lightgbm = LightGBMModel(self.lightgbm_config, self.evaluation_config)
        self.model_results.append(lightgbm.train(self.dataset, self.feature_columns))

        catboost = CatBoostModel(self.catboost_config, self.evaluation_config)
        self.model_results.append(catboost.train(self.dataset, self.feature_columns))
        return self.model_results

    def run_selection_pipeline(self) -> dict | None:
        if not self.model_results:
            return None
        selector = ModelSelector(self.model_selector_config, self.evaluation_config)
        self.final_model = selector.select(self.model_results, self.dataset)
        return self.final_model

    def run_test_pipeline(self) -> pd.DataFrame | None:
        if self.final_model is None:
            return None
        tester = HoldoutTester(self.evaluation_config)
        self.test_results = tester.test(self.model_results, self.final_model, self.dataset)
        return self.test_results