from util import *
from config import *
import concurrent.futures

def baseline_main():
    c_code = open(C_CODE_FILE, "r").read()
    designs = load_designs_from_pickle(PICKLE_FILE, 10) # top 10
    works = []
    for i, d in enumerate(designs):
        works.append(apply_design_to_code(WORK_DIR, c_code, d, i))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for curr_dir in works:
            futures.append(executor.submit(run_merlin_compile, curr_dir))
        for future in concurrent.futures.as_completed(futures):
            future.result()
            
if __name__ == "__main__":
    baseline_main()