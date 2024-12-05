import json
from config import *
from util import *
from idea0 import Idea0Explorer
from idea1 import Idea1Explorer

def llm_dse(c_code, config_file):
    logfile_path = WORK_DIR + "/log.txt"
    logfile = open(logfile_path, "a")
    curr_design: dict = get_default_design(config_file)
    explorer = Idea1Explorer(c_code, logfile)
    for i_steps in range(MAX_ITER):
        prompt_str = ""
        print("-"*80 + f"\nStarting iteration {i_steps}")
        print("-"*80 + f"\nStarting iteration {i_steps}", file=logfile)
        explorer.designs.append(curr_design)
        curr_dir = apply_design_to_code(WORK_DIR, c_code, curr_design, i_steps)
        if i_steps != 0: run_merlin_compile(curr_dir)
        if i_steps >= MAX_ITER - 1: break
        curr_design = explorer.explore(i_steps)
        assert isinstance(curr_design, dict), f"expecting dict, got {type(curr_design)}"

    logfile.close()

if __name__ == "__main__":
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code, CONFIG_FILE)
