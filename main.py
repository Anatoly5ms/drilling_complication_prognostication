from src.pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    training_pipeline = TrainingPipeline()
    tables = training_pipeline.run_data_pipeline()
    labels = training_pipeline.run_labeling_pipeline()
    dataset = training_pipeline.run_feature_pipeline()
    model_results = training_pipeline.run_model_pipeline()
    final_model = training_pipeline.run_selection_pipeline()
    test_results = training_pipeline.run_test_pipeline()


if __name__ == '__main__':
    main()