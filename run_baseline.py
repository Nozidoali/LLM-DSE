from util import *
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

BASELINE_DIR = './baseline'
BASELINE_WORK_DIR = '/scratch/alicewu/baseline-2'
DATA_DIR = './data/lad25'

idx = 0
datas = []

compile_args = []
for file in os.listdir(BASELINE_DIR):
    if file.endswith('.json'):
        design = json.load(open(f'{BASELINE_DIR}/{file}', 'r'))
        names = file.split('.')[0].split('-')
        kernel, shot, index, know, arbitrator = '-'.join(names[:-4]), names[-4], names[-3], names[-2], names[-1]
        if index != "2": continue
        c_code = open(f'{DATA_DIR}/{kernel}.c', 'r').read()
        print(f'Running {file}')
        idx += 1
        compile_args.append((BASELINE_WORK_DIR, c_code, design, idx))
with ThreadPoolExecutor(max_workers=15) as executor:   
    merlin_results = list(executor.map(lambda args: eval_design(*args), compile_args))
    
for i, merlin_result in enumerate(merlin_results):
    merlin_res, merlin_log = merlin_result
    datas.append({**design, **merlin_res, 'kernel': kernel, 'shot': shot, 'know': know, 'arbitrator': arbitrator, 'log': "\n".join(merlin_log)})
    pd.DataFrame(datas).to_csv(f'{BASELINE_DIR}/result.csv', index=False)