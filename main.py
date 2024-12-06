import json
from config import *
from util import *
from explorer import *

def llm_dse(c_code, config_file):
    curr_design: dict = get_default_design(config_file)
    explorer = Idea1Explorer(c_code)
    for i_steps in range(MAX_ITER):
        print("-"*80 + f"\nStarting iteration {i_steps}")
        explorer.designs.append(curr_design)
        curr_dir = apply_design_to_code(WORK_DIR, c_code, curr_design, i_steps)
        if i_steps >= 1 or not os.path.exists(os.path.join(curr_dir, "merlin.rpt")): run_merlin_compile(curr_dir)
        if i_steps >= MAX_ITER - 1: break
        curr_design = explorer.explore(i_steps)

if __name__ == "__main__":
    print_config()
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code, CONFIG_FILE)
