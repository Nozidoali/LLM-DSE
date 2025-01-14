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

def format_design(design: dict) -> str:
    return ", ".join([f"{k} = {v}" for k, v in design.items()])

def format_results(results: dict) -> str:
    if results == {} or "cycles" not in results or results["cycles"] == "": return "Compilation Timeout."
    return ", ".join([f"{k} = {v}" for k, v in results.items()])

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
    [os.mkdir(d) for d in [work_dir, curr_dir, curr_src_dir] if not os.path.exists(d)]
    os.makedirs(mcc_common_dir, exist_ok=True)
    c_path = os.path.join(curr_src_dir, f"{KERNEL_NAME}.c")
    curr_code: str = c_code
    for key, value in curr_design.items():
        curr_code = curr_code.replace("auto{" + key + "}", str(value))
    open(c_path, 'w').write(curr_code)
    open(curr_dir + "Makefile", 'w').write(MAKEFILE_STR)
    open(mcc_common_dir + "mcc_common.mk", 'w').write(MCC_COMMON_STR)
    time.sleep(5) # wait for the file to be written
    return curr_dir


def run_merlin_compile(make_dir: str) -> Tuple[Dict[str, str], List[str]]:
    merlin_rpt_file = os.path.join(make_dir, "merlin.rpt")
    merlin_log_file = os.path.join(make_dir, "merlin.log")
    start = time.time()
    try:
        process = subprocess.Popen(f"cd {make_dir} && make clean && rm -rf .merlin_prj 2>&1 > /dev/null && make mcc_estimate 2>&1 > /dev/null", shell=True, preexec_fn = os.setsid)
        process.wait(timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("Compilation Timeout. Killing the process group...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try: process.wait(5)
        except subprocess.TimeoutExpired: os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    elapsed = time.time() - start
    if os.path.exists(os.path.join(make_dir, ".merlin_prj")): subprocess.run(f"rm -rf {os.path.join(make_dir, '.merlin_prj')}", shell=True)
    if os.path.exists(os.path.join(make_dir, f"{KERNEL_NAME}.mco")): subprocess.run("rm -f " + os.path.join(make_dir, f"{KERNEL_NAME}.mco"), shell=True)
    subprocess.run(f"rm -f {os.path.join(make_dir, '*.zip')}", shell=True)
    time.sleep(10) # wait for the file to be written
    minutes, seconds = divmod(int(elapsed), 60)
    return {"compilation time": f"{minutes:02d}min {seconds:02d}sec", **parse_merlin_rpt(merlin_rpt_file)}, parse_merlin_log(merlin_log_file)


def eval_design(work_dir: str, c_code: str, curr_design: dict, idx: int) -> Tuple[Dict[str, str], List[str]]:
    if DATABASE_IS_VALID:
        import pandas as pd
        df = pd.read_csv(DATABASE_FILE)
        matched_results = df[(df[list(curr_design.keys())] == pd.Series(curr_design)).all(axis=1)]
        if len(matched_results) > 0:
            datas = matched_results.drop(columns=list(curr_design.keys())).to_dict(orient='records')
            merlin_results = {}
            for data in datas: merlin_results.update(data)
            print(f"INFO: loaded from database {json.dumps(merlin_results, indent=2)}\n\t design: {json.dumps(curr_design, indent=2)}")
            return merlin_results, [] # warnings are not available
    make_dir: str = apply_design_to_code(work_dir, c_code, curr_design, idx)
    merlin_results, merlin_log = run_merlin_compile(make_dir)
    print(f"INFO: havest after compilation {json.dumps(merlin_results, indent=2)}\n\t design: {json.dumps(curr_design, indent=2)}")
    return merlin_results, merlin_log

def get_openai_response(prompt, model=GPT_MODEL) -> str:
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

def handle_response_exceptions(func):
    def wrapper(response: str):
        try:
            return func(response)
        except Exception:
            print(f"WARNING: invalid response received {response}")
            traceback.print_exc()
            return None if func.__name__ == 'retrieve_index_from_response' else []
    return wrapper

@handle_response_exceptions
def retrieve_code_from_response(response: str) -> str:
    return response.replace("```c++", "").replace("```", "").strip()

@handle_response_exceptions
def retrieve_dict_from_response(response: str) -> dict:
    design = json.loads(re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)[0])
    return design

@handle_response_exceptions
def retrieve_list_from_response(response: str) -> List[dict]:
    return [json.loads(match) for match in re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)]

@handle_response_exceptions
def retrieve_index_from_response(response: str) -> int:
    return int(response.strip())

@handle_response_exceptions
def retrieve_indices_from_response(response: str) -> List[int]:
    return [int(x) for x in response.strip().split(",")]


def compile_best_design_prompt(c_code: str, exploration_history: list, best_design_history: list, best_design_info: dict) -> str:
    n_best_designs: int = min(NUM_BEST_DESIGNS, len(exploration_history))
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n``` with some pragma placeholders for high-level synthesis (HLS), your task is to choose the top {n_best_designs} best designs among the following options.",
        f"Here are the design spaces for the HLS pragmas:",
        *[f" {i}: {format_design(design)}. The results are: {format_results(hls_results)} and the remaining search space is {best_design_info[i]['remaining space']} out of {best_design_info[i]['total space']}."
          for i, (_, design, hls_results, _) in enumerate(exploration_history)],
        f"You are encouraged to choose the best designs that are not in the best design history.",
        *KNOWLEDGE_DATABASE['best_design'],
        f"Note that in the exploration history, we have chosen the following indices as the best designs:\n" + "\n ".join([str(idx) for idx in best_design_history]),
        f"This is because we are doing a design space exploration, and we want to find the design that can be further optimized.",
        f"You must skip the reasoning and output a list of integer values separated by ',' and the values should be in the range of {range(len(exploration_history))} representing the top {n_best_designs} best designs among the following options.",
    ])

def compile_warning_analysis_prompt(warnings: List[str], pragma_names: List[str]) -> str:
    return "\n".join([
        f"You must decide for each pragma below a list of warnings that you should consider when updating the pragma.",
        f"For example, if you have the following warning:",
        f"WARNING: [CGPIP-208] Coarse-grained pipelining NOT applied on loop 'Lx' (top.c:YY)",
        f"It means that the coarse-grained pipelining is not applied on loop 'Lx' at line YY.",
        f"In this example, you should consider this warning when updating the pipeline pragma for loop 'Lx', i.e., __PIPE__Lx.",
        f"Based on the HLS compilation log, there are some warnings that you should assign to pragmas:",
        "\n".join([f"\t{i}. {warning}" for i, warning in enumerate(warnings)]),
        f"The list of pragmas is:",
        "\n".join([f"\t{i}. {pragma_name}" for i, pragma_name in enumerate(pragma_names)]),
        f"You must output a JSON string with the pragma name as the key and the list of original warnings as the value.",
        f"You don't need to include all the warnings, only the ones that are relevant to a pragma.",
        f"Never output the reasoning and you must make sure the JSON string is valid.",
    ])

def compile_pragma_update_prompt(best_design: dict, hls_results: Dict[str, str], pragma_name: str, c_code: str, all_options: List[str], pragma_type: str, hls_warnings: List[str], exploration_history: Dict[str, str]) -> str:
    n_optimizations: int = min(NUM_OPTIMIZATIONS, len(all_options) - 1) if pragma_type != "pipeline" else 1
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n``` with some pragma placeholders for high-level synthesis (HLS), your task is to update the {pragma_type} pragma {pragma_name}.",
        f"You must choose {n_optimizations} values among {all_options} other than {best_design[pragma_name]} and values " + ", ".join([f"{k}" for k in exploration_history.keys()]) + ".",
        f"that can optimize the performance the most (reduce the cycle count) while keeping the resource utilization under 80% and the compilation time under {COMPILE_TIMEOUT_MINUTES} minutes.",
        f"Note that when: {format_design(best_design)}",
        ((f"We received the warning:\n" + "\n".join(hls_warnings)) if hls_warnings != [] else ""),
        (f"If the warning suggests the pragma is not applied due to dependency or other reasons, you could decide to skip the pragma update by outputting an empty string." if hls_warnings != [] else ""),
        f"The kernel's results after HLS synthesis are:\n {format_results(hls_results)}",
        "\n".join([f"and when {pragma_name} is {k}, the results are: {v}" for k, v in exploration_history.items()]),
        f"To better decide the {pragma_type} factor, here is some knowledge about {pragma_type} pragmas:",
        *KNOWLEDGE_DATABASE[pragma_type],
        f"You must skip the reasoning and only output at most {n_optimizations} separate JSON strings, i.e., ```json{{\"{pragma_name}\": value}}```, which holds {n_optimizations} different values."
    ])


def compile_arbitrator_prompt(best_design: dict, hls_results: Dict[str, str], list_of_warnings: List[str], pragma_updates: List[tuple], c_code: str) -> str:
    objective = "optimize clock cycles the most." if hls_results != {} else "reduce the resource utilization and the compilation time."
    n_designs: int = min(NUM_CHOSENS, len(pragma_updates))
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high-level synthesis (HLS), your task is to choose {n_designs} single updates from the following updates that {objective}",
        "\n".join([f"({i}): change {k} from {best_design[k]} to {v}" for i, (k, v) in enumerate(pragma_updates)]),
        f"The CURRENT DESIGN is: {format_design(best_design)}",
        f"The kernel's results after HLS synthesis are:\n {format_results(hls_results)}",
        f"The list of warnings is:\n" + "\n".join(list_of_warnings),
        f"To better understand the problem,",
        *KNOWLEDGE_DATABASE['general'],
        *KNOWLEDGE_DATABASE['parallel'],
        *KNOWLEDGE_DATABASE['pipeline'],
        *KNOWLEDGE_DATABASE['tile'],
        f"To make a better decision,", *KNOWLEDGE_DATABASE['arbitrator'],
        f"Make the update to the current design and you must output the new design with the following pragma's values: " + ",".join(best_design.keys()) + " as a JSON string, i.e., can be represented as ```json{\"pragma1\": value1, \"pragma2\": value2, ...}```",
        f"Note that in total you should only output {n_designs} separate JSON strings, which means {n_designs} designs. For each one of them, the new design should only have one pragma different from the original one (CURRENT DESIGN).",
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
    return [l.strip() for l in open(input_file, "r").readlines() if "WARNING" in l]


def _parse_options(pragma_name: str, options: str) -> List[str]:
    option_list = eval(re.search(r'\[([^\[\]]+)\]', options).group(0))
    if "PARA" in pragma_name: return [str(x) for x in option_list if option_list[-1] % x == 0]
    return [str(x) for x in option_list]

def compile_design_space(config_file: str) -> dict:
    config_dict = json.load(open(config_file, "r"))["design-space.definition"]
    return {p: _parse_options(p, config_dict[p]['options']) for p in config_dict}

