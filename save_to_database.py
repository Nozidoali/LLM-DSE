import os
import pandas as pd
import re

result_folders = [
    # 'results0102',
    # 'results0103',
    # 'results0104',
    # 'results0108',
    # 'results0109',
    # 'results0110',
    # 'results0112',
    'results0114',
]
database_folder = './data/compilation_results'

pragma_pattern = re.compile(r'__(PARA|TILE|PIPE)__')
pattern = re.compile(r'compilation time|cycles|lut utilization|FF utilization|BRAM utilization|DSP utilization|URAM utilization|__(PARA|TILE|PIPE)__')

if __name__ == '__main__':
    dfs = {}
    for folder in result_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                bmark = filename.split('.')[0]
                if bmark not in dfs: dfs[bmark] = pd.read_csv(os.path.join(folder, filename))
                else: dfs[bmark] = pd.concat([dfs[bmark], pd.read_csv(os.path.join(folder, filename))])
    for bmark, df in dfs.items():
        if os.path.exists(os.path.join(database_folder, f'{bmark}.csv')):
            df = pd.concat([df, pd.read_csv(os.path.join(database_folder, f'{bmark}.csv'))])
        # drop the column called step
        df = df.drop(columns=['step'])
        pragma_names = [col for col in df.columns if pragma_pattern.search(col)]
        df = df[[col for col in df.columns if pattern.search(col)]]
        df = df[~(df['cycles'].isna() & ~df['compilation time'].isin(['40min 00sec', '60min 00sec', '80min 00sec']))]
        df = df.dropna(subset=pragma_names)        
        df = df.drop_duplicates()
        df.to_csv(os.path.join(database_folder, f'{bmark}.csv'), index=False)
