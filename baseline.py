from util import *
from config import *

def baseline_main():
    c_code = open(C_CODE_FILE, "r").read()
    designs = load_designs_from_pickle(PICKLE_FILE, 10) # top 10
    for i, d in enumerate(designs):
        curr_dir = apply_design_to_code(WORK_DIR, c_code, d, i)
        run_merlin_compile(curr_dir)
    
if __name__ == "__main__":
    baseline_main()