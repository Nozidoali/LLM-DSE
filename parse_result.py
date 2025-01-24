from util import *
import pandas as pd

DATA_DIR = './data/lad25'
BASELINE_WORK_DIR = '/home/alicewu/lad25/cs259-llm-dse/baseline'
BASELINE_DIR = './baseline'
BENCHMARK_SUITES = ["3mm", "atax-medium", "covariance", "fdtd-2d", "gemm-p", "gemver-medium", "jacobi-2d", "symm-opt", "syr2k", "trmm-opt"]

def find_kernel(c_file:str):
    lines = open(c_file, 'r').readlines()
    for i, line in enumerate(lines):
        match = re.search(r'void\s+kernel_(\w+)\s*\(', line)
        if match:
            return match.group(1).replace("_", "-")

def parse_design(kernel_name:str, c_file:str):
    design = {}
    for file in os.listdir(DATA_DIR):
        if file.startswith(kernel_name) and file.endswith('.c'):
            c_code_lines = open(f'{DATA_DIR}/{file}', 'r').readlines()
            break
    design_lines = open(c_file, 'r').readlines()
    for c_line, d_line in zip(c_code_lines, design_lines):
        if "PIPELINE" in c_line:
            key = re.search(r"auto\{(.*?)\}", c_line).group(1)
            match = re.search(r"PIPELINE\s+(\w+)", d_line)
            if match:
                value = match.group(1) 
            else:
                value = ""
            design[key] = value
        elif "TILE" in c_line:
            key = re.search(r"auto\{(.*?)\}", c_line).group(1)
            value = re.search(r"FACTOR\s*=\s*(\d+)", d_line).group(1)
            design[key] = value
        elif "PARALLEL" in c_line:
            key = re.search(r"auto\{(.*?)\}", c_line).group(1)
            value = re.search(r"FACTOR\s*=\s*(\d+)", d_line).group(1)
            design[key] = value
    return design

def match_design(design:dict, kernel_name:str):
    for file in os.listdir(BASELINE_DIR):
        if file.startswith(kernel_name) and file.endswith('.json'):
            save_design = json.load(open(f'{BASELINE_DIR}/{file}', 'r'))
            save_design = {key: str(value) for key, value in save_design.items()}
            if save_design == design:
                names = file.split('.')[0].split('-')
                kernel, shot, index, know, arbitrator = '-'.join(names[:-4]), names[-4], names[-3], names[-2], names[-1]
                print(kernel, shot, index, know, arbitrator)
                return kernel, shot, index, know, arbitrator

def load():
    datas = {}

    for j in range(8):
        CURRENT_DIR = f'{BASELINE_WORK_DIR}/baseline-{j}'
        for i in range(1, 61, 1):
            merlin_rpt_file = f'{CURRENT_DIR}/{i}/merlin.rpt'
            merlin_log_file = f'{CURRENT_DIR}/{i}/merlin.log'
            merlin_rpt = parse_merlin_rpt(merlin_rpt_file)
            merlin_log = parse_merlin_log(merlin_log_file)
            if os.path.exists(f'{CURRENT_DIR}/{i}/src'):
                c_file = f'{CURRENT_DIR}/{i}/src/top.c'
            else:
                c_file = f'{CURRENT_DIR}/{i}/top.c'
            kernel_name = find_kernel(c_file)
            print(kernel_name)
            design = parse_design(kernel_name, c_file)
            print(design)
            kernel, shot, index, know, arbitrator = match_design(design, kernel_name)
            if kernel not in datas:
                datas[kernel] = []
            datas[kernel].append({**design, **merlin_rpt, 'kernel': kernel, 'shot': shot, 'know': know, 'arbitrator': arbitrator})
            if not os.path.exists(f'{BASELINE_DIR}/{kernel}'):
                os.makedirs(f'{BASELINE_DIR}/{kernel}')
            pd.DataFrame(datas[kernel]).to_csv(f'{BASELINE_DIR}/{kernel}/result.csv')

def extract_parathesis(s):
    return int(re.search(r'\((.*?)\)', s).group(1).replace("~", "").replace("%", ""))/100 if isinstance(s, str) and "(" in s else INT_MAX
def exclude_parathesis(s):
    return int(s.split("(")[0].strip()) if isinstance(s, str) and "(" in s else INT_MAX

INT_MAX = 2**31 - 1

def process_result():
    results = []
    for bmark in BENCHMARK_SUITES:
        csv_file = f"{BASELINE_WORK_DIR}/{bmark}/result.csv"
        df = pd.read_csv(csv_file)
        for shot in ["zero", "one"]:
            for know in [False, True]:
                for arbitrator in [False, True]:
                    if know == False and arbitrator == True: continue
                    df_ = df[(df['shot'] == shot) & (df['know'] == know) & (df['arbitrator'] == arbitrator)].copy()
                    df_["cycles"] = df_["cycles"].apply(exclude_parathesis)
                    for k in ["lut utilization", "DSP utilization", "FF utilization", "BRAM utilization", "URAM utilization"]:
                        df_[k] = df_[k].apply(extract_parathesis)
                    df_["max util."] = df_[["lut utilization", "DSP utilization", "FF utilization", "BRAM utilization", "URAM utilization"]].max(axis=1)
                    df_["perf"] = df_["cycles"]
                    df_.loc[df_["max util."] > 0.8, "perf"] = INT_MAX
                    best_perf = df_["perf"].min()
                    print(f"{bmark},{shot},{know},{arbitrator},{best_perf}")
                    results.append({
                        "bmark": bmark,
                        "shot": shot,
                        "know": know,
                        "arbitrator": arbitrator,
                        "best_perf": best_perf
                    })
    pd.DataFrame(results).to_csv(f'{BASELINE_WORK_DIR}/best.csv', index=False)


process_result()


pragma_pattern = re.compile(r'__(PARA|TILE|PIPE)__')
pattern = re.compile(r'compilation time|cycles|lut utilization|FF utilization|BRAM utilization|DSP utilization|URAM utilization|__(PARA|TILE|PIPE)__')

for i in range(1, 61, 1):
    merlin_rpt_file = f'{BASELINE_WORK_DIR}/{i}/merlin.rpt'
    merlin_log_file = f'{BASELINE_WORK_DIR}/{i}/merlin.log'
    merlin_rpt = parse_merlin_rpt(merlin_rpt_file)
    merlin_log = parse_merlin_log(merlin_log_file)
    c_file = f'{BASELINE_WORK_DIR}/{i}/src/top.c'
    kernel_name = find_kernel(c_file)
    print(kernel_name)
    design = parse_design(kernel_name, c_file)
    print(design)
    kernel, shot, index, know, arbitrator = match_design(design, kernel_name)
    assert(index!=7)
    if kernel not in datas:
        datas[kernel] = []
    datas[kernel].append({**design, **merlin_rpt, 'kernel': kernel, 'shot': shot, 'know': know, 'arbitrator': arbitrator})
    if not os.path.exists(f'{BASELINE_DIR}/{kernel}'):
        os.makedirs(f'{BASELINE_DIR}/{kernel}')
    pd.DataFrame(datas[kernel]).to_csv(f'{BASELINE_DIR}/{kernel}/result-7.csv', index=False)



            




