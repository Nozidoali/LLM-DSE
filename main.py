import json
from config import *
from util import *
from explorer import *
from concurrent.futures import ThreadPoolExecutor

def llm_dse(c_code, config_file):
    curr_design: dict = get_default_design(config_file)
    curr_design_list = [curr_design]
    explorer = Explorer(c_code)
    i_steps = 0
    while i_steps < MAX_ITER:
        print("-"*80 + f"\nStarting iteration {i_steps}")
        curr_dirs = []
        for i, curr_design in enumerate(curr_design_list):
            curr_dir = apply_design_to_code(WORK_DIR, explorer.c_code, curr_design, i_steps + i)
            curr_dirs.append(curr_dir)
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            merlin_results = list(executor.map(run_merlin_compile, curr_dirs))
        # if i_steps >= 1 or not os.path.exists(os.path.join(curr_dir, "merlin.rpt")):
        #     merlin_res = run_merlin_compile(curr_dir)
        # else:
        #     merlin_res = parse_merlin_rpt(os.path.join(curr_dir, "merlin.rpt")), parse_merlin_log(os.path.join(curr_dir, "merlin.log"))
        for i, (design, merlin_res) in enumerate(zip(curr_design_list, merlin_results)):
            explorer.record_history(i_steps + i, design, *merlin_res)
        i_steps += len(curr_design_list)
        if i_steps >= MAX_ITER - 1: break
        curr_design_list = explorer.explore()

if __name__ == "__main__":
    print_config()
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code, CONFIG_FILE)
