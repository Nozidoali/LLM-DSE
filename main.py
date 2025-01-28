import json, time
from config import *
from util import *
from explorer import *
from concurrent.futures import ThreadPoolExecutor

def llm_dse(c_code):
    designs = [(None, get_default_design(CONFIG_FILE))]
    explorer = Explorer(c_code, get_default_design(CONFIG_FILE).keys())
    while explorer.proceed():
        compile_args = [(WORK_DIR, explorer.c_code, design, explorer.i_steps + i) for i, (_, design) in enumerate(designs)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            merlin_results = list(executor.map(lambda args: eval_design(*args), compile_args))
        for i, merlin_result in enumerate(merlin_results):
            prev_design, design = designs[i]
            prev_hls_results, prev_hls_warnings = explorer.load_results(prev_design)
            curr_hls_results, curr_hls_warnings = merlin_result
            reflection = explorer.self_reflection(prev_design, design, prev_hls_results, prev_hls_warnings, curr_hls_results, curr_hls_warnings, explorer.get_index(prev_design), explorer.i_steps)
            pragma_warnings = explorer.analyze_warnings(curr_hls_warnings)
            explorer.record_history(design, curr_hls_results, pragma_warnings, reflection)
        designs = [(design, {**design, pragma_name: pragma_value}) for design, pragma_name, pragma_value in explorer.explore()]

if __name__ == "__main__":
    print_config()
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code)
