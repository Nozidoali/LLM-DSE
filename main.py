import json
from config import *
from util import *
from explorer import *
from concurrent.futures import ThreadPoolExecutor

def llm_dse(c_code, config_file):
    curr_design_list = [get_default_design(config_file)]
    explorer = Explorer(c_code)
    i_steps = 0
    while i_steps < MAX_ITER:
        print("-"*80 + f"\nStarting iteration {i_steps}")
        compile_args = [(WORK_DIR, explorer.c_code, design, i_steps + i) for i, design in enumerate(curr_design_list)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            merlin_results = list(executor.map(lambda args: eval_design(*args), compile_args))
        for i, (design, merlin_res) in enumerate(zip(curr_design_list, merlin_results)):
            explorer.record_history(i_steps + i, design, *merlin_res)
        i_steps += len(curr_design_list)
        if i_steps >= MAX_ITER - 1: break
        curr_design_list = explorer.explore()

if __name__ == "__main__":
    print_config()
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code, CONFIG_FILE)
