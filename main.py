import json, time
from config import *
from util import *
from explorer import *
from concurrent.futures import ThreadPoolExecutor

def llm_dse(c_code):
    designs = [(None, get_default_design(CONFIG_FILE))]
    explorer = Explorer(c_code, get_default_design(CONFIG_FILE).keys())
    ticks = [time.time()]
    open(TIME_LOGFILE, 'w').close()
    i_steps, i_iter = 0, 0
    while i_iter < MAX_ITER:
        i_iter += 1
        print("-"*80 + f"\nStarting iteration {i_steps}")
        if len(designs) == 0: break
        compile_args = [(WORK_DIR, explorer.c_code, design, i_steps + i) for i, (_, design) in enumerate(designs)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            merlin_results = list(executor.map(lambda args: eval_design(*args), compile_args))
        for i, merlin_result in enumerate(merlin_results):
            prev_design, design = designs[i]
            prev_hls_results, prev_hls_warnings = explorer.load_results(prev_design)
            curr_hls_results, curr_hls_warnings = merlin_result
            reflection = explorer.self_reflection(prev_design, design, prev_hls_results, prev_hls_warnings, curr_hls_results, curr_hls_warnings, explorer.get_index(prev_design), i_steps)
            reflection = "useful" if i_steps == 0 else reflection
            pragma_warnings = explorer.analyze_warnings(curr_hls_warnings)
            explorer.record_history(i_iter, i_steps, time.time()-ticks[0], design, curr_hls_results, pragma_warnings, reflection)
            i_steps += 1
        ticks.append(time.time())
        open(TIME_LOGFILE, 'a').write(f'Iteration {len(ticks)-1}, Total runtime: {ticks[-1] - ticks[0]}, Iteration runtime: {ticks[-1] - ticks[-2]}\n')
        designs = [(design, {**design, pragma_name: pragma_value}) for design, pragma_name, pragma_value in explorer.explore()]

if __name__ == "__main__":
    print_config()
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code)
