import json, time
from config import *
from util import *
from explorer import *
from concurrent.futures import ThreadPoolExecutor

def llm_dse(c_code):
    prev_design_list, curr_design_list = [None], [get_default_design(CONFIG_FILE)]
    explorer = Explorer(c_code, curr_design_list[0].keys())
    ticks = [time.time()]
    open(TIME_LOGFILE, 'w').close()
    i_steps = 0
    while i_steps < MAX_ITER:
        print("-"*80 + f"\nStarting iteration {i_steps}")
        compile_args = [(WORK_DIR, explorer.c_code, design, i_steps + i) for i, design in enumerate(curr_design_list)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            merlin_results = list(executor.map(lambda args: eval_design(*args), compile_args))
        for i, (prev_design, design, merlin_res) in enumerate(zip(prev_design_list, curr_design_list, merlin_results)):
            prev_hls_results, prev_hls_warnings = explorer.load_results(prev_design)
            hls_results, hls_warnings = merlin_res
            pragma_warnings = {}
            if hls_warnings:
                warning_analysis_prompt = compile_warning_analysis_prompt(hls_warnings, explorer.pragma_names)
                pragma_warnings = retrieve_dict_from_response(get_openai_response(warning_analysis_prompt))
            explorer.self_reflection(prev_design, design, prev_hls_results, prev_hls_warnings, hls_results, hls_warnings)
            explorer.record_history(i_steps + i, design, hls_results, pragma_warnings)
        i_steps += len(curr_design_list)
        if i_steps >= MAX_ITER - 1: break
        ticks.append(time.time())
        open(TIME_LOGFILE, 'a').write(f'Iteration {len(ticks)-1}, Total runtime: {ticks[-1] - ticks[0]}, Iteration runtime: {ticks[-1] - ticks[-2]}\n')
        prev_design_list, curr_design_list = zip(*explorer.explore())

if __name__ == "__main__":
    print_config()
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code)
