import pandas as pd

target = [
    'train_data',
    'test_data',
]

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in target:
    (pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8"))\
        .to_feather('./data/input/' + t + '.feather')
