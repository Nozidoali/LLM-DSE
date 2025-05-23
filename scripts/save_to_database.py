import os
import pandas as pd
import re

result_folders = [
    'exp_arb_1',
    'exp_arb_2',
    'exp_heuristic',
    'exp_jacobi',
    'exp_opt',
    'exp_opt_arb',
    'exp_opt_arb_bes',
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
        for col in pragma_names:
            if 'PARA' in col or 'TILE' in col:
                df[col] = df[col].astype(int)
        df.to_csv(os.path.join(database_folder, f'{bmark}.csv'), index=False)
