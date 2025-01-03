import os
import json
import re
import subprocess
import openai
import traceback
import pickle
import torch
import logging
from typing import List, Dict, Union, Optional, Tuple
from config import *
import signal
import time
from datetime import timedelta

def get_default_design(ds_config_file: str) -> dict:
    config_dict = json.load(open(ds_config_file, "r"))["design-space.definition"]
    return {p: config_dict[p]["default"] for p in config_dict}

def designs_are_adjacent(design1: dict, design2: dict) -> bool:
    return sum([design1[k] != design2[k] for k in design1.keys()]) == 1

def load_designs_from_pickle(pickle_file: str, n_best: int = 10) -> List[dict]:
    results = [d for _, d in pickle.load(open(pickle_file, "rb")).items()]
    selected = sorted(results, key=lambda x: x.perf, reverse=True)[:n_best]
    print(f"INFO: selected {len(selected)} designs from {len(results)} designs")
    return [{k: (v.item() if torch.is_tensor(v) else v) for k, v in d.point.items()} for d in selected]

def apply_design_to_code(work_dir: str, c_code: str, curr_design: dict, idx: int) -> str:
    curr_dir = os.path.join(work_dir, f"{idx}/")
    curr_src_dir = os.path.join(curr_dir, "src/")
    mcc_common_dir = os.path.join(work_dir, "mcc_common/")
    [os.mkdir(d) for d in [work_dir, curr_dir, curr_src_dir, mcc_common_dir] if not os.path.exists(d)]
    c_path = os.path.join(curr_src_dir, f"{KERNEL_NAME}.c")
    curr_code: str = c_code
    for key, value in curr_design.items():
        curr_code = curr_code.replace("auto{" + key + "}", str(value))
    open(c_path, 'w').write(curr_code)
    open(curr_dir + "Makefile", 'w').write(MAKEFILE_STR)
    open(mcc_common_dir + "mcc_common.mk", 'w').write(MCC_COMMON_STR)
    return curr_dir


def run_merlin_compile(make_dir: str) -> Tuple[Dict[str, str], List[str]]:
    merlin_rpt_file = os.path.join(make_dir, "merlin.rpt")
    merlin_log_file = os.path.join(make_dir, "merlin.log")
    start = time.time()
    try:
        process = subprocess.Popen(f"cd {make_dir} && make clean && rm -rf .merlin_prj && make mcc_estimate 2>&1 > /dev/null", shell=True, preexec_fn = os.setsid)
        process.wait(timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("Compilation Timeout. Killing the process group...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try: process.wait(5)
        except subprocess.TimeoutExpired: os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)
    return {"compilation time": f"{minutes:02d}min {seconds:02d}sec", **parse_merlin_rpt(merlin_rpt_file)}, parse_merlin_log(merlin_log_file)


def get_openai_response(prompt, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in design HLS codes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10000,  # Set the largest token numbers
        temperature=0.7,  # Control the randomness of the generative result
    )
    open(OPENAI_LOGFILE, "a").write("\n" + "=" * 80 + "\n" + prompt + "\n" + "-" * 80 + "\n" + response.choices[0].message.content)
    return(response.choices[0].message.content)

def retrieve_code_from_response(response: str) -> str:
    try:
        return response.replace("```c++", "").replace("```", "").strip()
    except Exception:
        print(f"WARNING: invalid response received {response}"); traceback.print_exc()
        return ""

def retrieve_dict_from_response(response: str) -> dict:
    try:
        _response = response.replace("```json", "").replace("```", "").replace("\n", " ").strip()
        design = json.loads("{"+re.findall(r'\{(.*?)\}', _response)[0]+"}")
        return design
    except Exception:
        print(f"WARNING: invalid response received {response}"); traceback.print_exc()
        return {}

def retrieve_index_from_response(response: str) -> int:
    try:
        return int(response.strip())
    except Exception:
        print(f"WARNING: invalid response received {response}"); traceback.print_exc()
        return None
    
KNOWLEDGE_DATABASE = {
    'general': [
        f"Here are some knowledge about the HLS pragmas you are encountering:",
        f"  (1) The pragmas only affect the next for loop after the pragma.",
        f"  (2) The pragmas are __PARA__LX, __PIPE__LX, and __TILE__LX, where LX is the loop name and X is an integer.",
    ],
    'parallel': [
        f"Here are some knowledge about the __PARA__LX pragma:",
        f"  (1) Parallel pragram will parallelize the first for loop in the c code under __PARA__.",
        f"  (2) Increasing the parallel factor will increase the resource utilization but improve the performance and decease the number of cycles (which is one of your target).",
        f"  (3) Increasing parallel factor roughly linearly increase the resource utilization within the loop it applies on, so you may scale the factor with respect to the ratio between current utilization with the 80% budget.",
        f"  (4) Increasing the parallel factor will also increase the compilation time, you must decrease the parallel factor if you received the compilation timeout.", 
        f"  (5) The compilation time is positively proportional to the parallel factor, you must choose the parallel factor such that the compilation time is under {COMPILE_TIMEOUT_MINUTES} minutes.",
    ], 
    'tile': [
        f"Here are some knowledge about the __TILE__LX pragma:",
        f"  (1) Tile pragma will tile the first for loop in the c code under __TILE__.",
        f"  (2) Increasing the tile factor will reduce the memory transfer cycles because it will restrict the memory transfer.",
    ],
    'pipeline': [
        f"Here are some knowledge about the __PIPE__LX pragma:",
        f"  (1) Pipeline pragma will affect MULTIPLE loops under __PIPE__.",
        f"  (2) The flatten option will unroll all the for loops (which means putting __PARA__ equals to the loop bound in the for loop) under this pragma.",
        f"  (3) Turning off the pipeline will not apply any pipelining, which is useful when you get compilation timeout in the report.",
        f"  (4) Choosing the empty string means coars-grained pipelining, which is useful when you believe the loop inside it has fewer loop-carried dependencies.",
    ],
    'arbitrator': [
        f"Here are some information about the preference:",
        f"  (1) You should prioritize optimizing the __PARA__ pragma first, as it affect the performance the most.",
        f"  (2) If you think all the parallel factors are already optimal, you consider pipeline as the secondary choice. When doing so, you must remember that the pipeline pragma will affect MULTIPLE loops. The flatten option will unroll all the for loops under this pragma. Turning off the pipeline will not apply any pipelining, which is useful when you get compilation timeout in the report.",
        f"  (3) If you think all the parallel factors are already optimal, and the pipeline pragma is already optimal, you can consider the tile pragma. The tile pragma will tile the first for loop in the c code under __TILE__.",
        f"  (4) By default, setting __TILE__ to 1 is perferable.",
        f"  (5) By default, setting __PIPE__ to 1 is perferable.",
    ]
}


def format_design(design: dict) -> str:
    return ", ".join([f"{k} is {v}" for k, v in design.items()])

def format_results(results: dict) -> str:
    if results == {}: return "Compilation Timeout."
    return ", ".join([f"{k} is {v}" for k, v in results.items()])

def rewrite_c_code(c_code: str) -> str:
    code_rewrite_prompt = "\n".join([
        f"Given the following C code:\n ```c++ \n{c_code}\n```",
        f"You must label each for loop with the corresponding pragma for high level synthesis (HLS).",
        f"Note that", *KNOWLEDGE_DATABASE['general'],
        """
        For example, if you have a for loop like this:
        ```c++
        #pragma HLS __PARA__L0
        for (int i = 0; i < N; i++) {
            // loop body
        }
        ```
        You should label it as follows:",
        ```c++
        #pragma HLS __PARA__L0
        L0: for (int i = 0; i < N; i++) {
            // loop body
        }
        ```
        """,
        f"You never modify the loop body and the functionality of the c code, only label the for loop with the pragma.",
        f"You must output the C code in a code block. You never output reasoning.",
    ])
    return retrieve_code_from_response(get_openai_response(code_rewrite_prompt))


def compile_best_design_prompt(c_code: str, exploration_history: list) -> str:
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n``` with some pragma placeholders for high level synthesis (HLS), your task is to choose the best design among the following options.",
        f"Here are the design space for the HLS pragmas:",
        *[f" {i}: {format_design(design)}. The results are: {format_results(hls_results)}" for i, design, hls_results, _ in exploration_history],
        f"A design is better if it has lower cycle count and resource utilization under 80%.",
        f"When the cycle count is the same, you should choose the design with lower resource utilization.",
        f"Note that the resource utilization is calculated by the max of LUT, FF, BRAM, DSP, and URAM utilization.",
        f"When the performance are similar, you should choose the design with more room for improvement.",
        f"This is because we are doing a design space exploration, and we want to find the design that can be further optimized.",
        f"You never output the reasoning, only the index of the best design.",
        f"You must output only an integer value between {range(len(exploration_history))} representing the best design among the following options.",
    ])
    

def compile_warning_analysis_prompt(warnings: List[str], pragma_names: List[str]) -> str:
    return "\n".join([
        f"Based on the HLS compilation log, there are some warnings that you should be aware of:",
        *warnings,
        f"You must decide for each pragma below a list of warnings that you should consider when updating the pragma.",
        *pragma_names,
        f"For example, if you have the following warnings:",
        f"WARNING: [CGPIP-208] Coarse-grained pipelining NOT applied on loop 'L0' (top.c:33)",
        f"It means that the coarse-grained pipelining is not applied on loop 'L0' at line 33.",
        f"You should consider this warning when updating the pipeline pragma for loop 'L0', i.e., __PIPE__L0.",
        f"You must output a JSON string with the pragma name as the key and the list of original warnings as the value.",
        f"You don't need to include all the warnings, only the ones that are relevant to the pragma.",
        f"Never output the reasoning and you must make sure the JSON string is valid.",
    ])


def compile_pragma_update_prompt(best_design: dict, hls_results: Dict[str, str], pragma_name: str, c_code: str, all_options: List[str], pragma_type: str, hls_warnings: List[str], exploration_history: Dict[str, str]) -> str:
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n``` with some pragma placeholders for high level synthesis (HLS), your task is to update the {pragma_type} pragma {pragma_name}.",
        f"You must choose one and only one value among {all_options} other than {best_design[pragma_name]} that can optimize the performance the most (reduce the cycle count) while keeping the resource utilization under 80% and the compilation time under {COMPILE_TIMEOUT_MINUTES} minutes.",
        f"Note that when: {format_design(best_design)}",
        (f"We received the warning:\n" + "\n".join(hls_warnings) if hls_warnings != [] else ""),
        f"The kernel's results after HLS synthesis are:\n {format_results(hls_results)}", 
        "\n".join([f"and when {pragma_name} is {k}, the results are: {v}" for k, v in exploration_history.items()]),
        f"To better decide the {pragma_type} factor, here are some knowledge about {pragma_type} pragmas:",
        *KNOWLEDGE_DATABASE[pragma_type],
        f"You must skip the reasoning and only output in JSON format string, i.e., {{{pragma_name}: value}}"
    ])
    

def compile_arbitrator_prompt(best_design: dict, hls_results: Dict[str, str], pragma_updates: List[tuple], c_code: str) -> str:
    objective = "optimize clock cycles the most." if hls_results != {} else "reduce the resource utilization and the compilation time."
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high level synthesis (HLS), your task is to choose one of the following updates that {objective}",
        "\n".join([f"({i}): change {k} from {best_design[k]} to {v}" for i, (k, v) in enumerate(pragma_updates)]),
        f"Note that when: {format_design(best_design)}",
        f"The kernel's results after HLS synthesis are:\n {format_results(hls_results)}",
        f"To better understand the problem, ", 
        *KNOWLEDGE_DATABASE['general'],
        *KNOWLEDGE_DATABASE['parallel'],
        *KNOWLEDGE_DATABASE['pipeline'],
        *KNOWLEDGE_DATABASE['tile'],
        f"To make better decision,", *KNOWLEDGE_DATABASE['arbitrator'],
        f"Make the update to the current design and output only the new pragma design for the keys: " + ",".join(best_design.keys()) + "as a JSON string. i.e., can be represented as {\"pragma1\": value1, \"pragma2\": value2, ...}",
    ])
    

def parse_merlin_rpt(merlin_rpt: str) -> Dict[str, str]:
    try:
        lines = open(merlin_rpt, "r").readlines()
        target_line_idx = [i for i, l in enumerate(lines) if "Estimated Frequency" in l]
        util_values = lines[target_line_idx[0]+4].split("|")[2:]
        util_keys = ['cycles', 'lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']
        return {util_keys[i]: util_values[i] for i in range(6)}
    except:
        return {}


def parse_merlin_log(input_file) -> List[str]:
    if not os.path.exists(input_file): return []
    return [l for l in open(input_file, "r").readlines() if "WARNING" in l]


def _parse_options(pragma_name: str, options: str) -> List[str]:
    option_list = eval(re.search(r'\[([^\[\]]+)\]', options).group(0))
    if "PARA" in pragma_name: return [str(x) for x in option_list if option_list[-1] % x == 0]
    return [str(x) for x in option_list]

def compile_design_space(config_file: str) -> dict:
    config_dict = json.load(open(config_file, "r"))["design-space.definition"]
    return {p: _parse_options(p, config_dict[p]['options']) for p in config_dict}

