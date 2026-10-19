import pandas as pd
from src.configs.data_config import get_data_config
from src.configs.labeler_config import get_labeler_config
from src.configs.target_config import get_target_config
from src.components.data_builder import DataBuilder
from src.components.labeler import Labeler
from src.utils.data_io import resolve_path


def main() -> None:
    pd.set_option('display.max_colwidth', 300)

    tables = DataBuilder(get_data_config()).build_tables()
    labeler = Labeler(get_labeler_config(), get_target_config(), tables)
    labels = labeler.build_labels()
    events = labels['events']

    columns = ['complication_type', 'nameWellbore', 'md_event', 'n_operations', 'sources',
               'in_drilled_range', 'comments_sample']

    for complication_type, group in events.groupby('complication_type'):
        print(f'\n===== {complication_type}: {len(group)} событий =====')
        print(group[columns[1:]].to_string())

    review_path = resolve_path('static/interim/labels/events_review.csv')
    review_path.parent.mkdir(parents=True, exist_ok=True)
    events[columns].to_csv(review_path, index=False)
    print(f'\nСохранено для проверки: {review_path}')


if __name__ == '__main__':
    main()