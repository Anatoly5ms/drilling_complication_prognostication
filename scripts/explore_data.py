from src.configs.data_config import get_data_config
from src.components.data_builder import DataBuilder
import pandas as pd

tables = DataBuilder(get_data_config()).build_tables()
act = tables['activity']
print(act['proprietaryCode'].value_counts())
# act['proprietaryCode'].value_counts().to_csv('/Users/anatoliigrigorovich/Desktop/proprietaryCode.csv')
pattern = r'\b([A-Z][a-zæøå]+)\s+(Fm|Formation|Gp|Group)\b'
mentions = act['comments'].str.extractall(pattern)
print(mentions[0].value_counts().head(30))
print(act.loc[act['comments'].str.contains(pattern, na=False), 'nameWellbore'].nunique())
print(act['state'].value_counts())