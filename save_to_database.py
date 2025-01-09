import os
import pandas as pd

result_folders = [
    'results0102',
    'results0103',
    'results0104',
]
database_folder = './data/compilation_results'

if __name__ == '__main__':
    dfs = {}
    for folder in result_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                bmark = filename.split('.')[0]
                if bmark not in dfs: dfs[bmark] = pd.read_csv(os.path.join(folder, filename))
                else: dfs[bmark] = pd.concat([dfs[bmark], pd.read_csv(os.path.join(folder, filename))])
    for bmark, df in dfs.items():
        df = df.drop_duplicates()
        df.to_csv(os.path.join(database_folder, f'{bmark}.csv'), index=False)
