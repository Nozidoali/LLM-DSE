import os
import json
import re
import subprocess
import openai
import traceback
import pickle
import torch
import logging
from typing import List, Dict, Union, Optional
from config import *
import signal

def get_default_design(ds_config_file: str) -> dict:
    config_dict = json.load(open(ds_config_file, "r"))["design-space.definition"]
    return {p: config_dict[p]["default"] for p in config_dict}

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


def run_merlin_compile(make_dir: str) -> None:
    merlin_rpt_file = os.path.join(make_dir, "merlin.rpt")
    if DEBUG: subprocess.run(f"echo hi > {merlin_rpt_file}", shell=True)
    else: 
        try:
            process = subprocess.Popen(f"cd {make_dir} && make clean && rm -rf .merlin_prj && make mcc_estimate > /dev/null", shell=True, preexec_fn = os.setsid)
            process.wait(timeout=COMPILE_TIMEOUT)
        except subprocess.TimeoutExpired:
            print("Compilation Timeout. Killing the process group...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                print("Process gorup killed.")

def get_openai_response(prompt, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in design HLS codes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Set the largest token numbers
        temperature=0.7,  # Control the randomness of the generative result
    )
    open(OPENAI_LOGFILE, "a").write("\n" + "=" * 80 + "\n" + prompt + "\n" + "-" * 80 + "\n" + response.choices[0].message.content)
    return(response.choices[0].message.content)

def get_openai_response_o1(prompt, model="o1-mini"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "You are an expert in design HLS codes."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5000,  # Set the largest token numbers
    )
    print(response)
    return(response.choices[0].message.content)

def retrieve_design_from_response(response: str) -> dict:
    try:
        _response = response.replace("```json", "").replace("```", "").replace("\n", " ").strip()
        print(f"INFO: response received {_response}")
        matches = re.findall(r'\{(.*?)\}', _response)
        print(f"INFO: matches {matches}")
        design = json.loads("{"+matches[0]+"}")
        return design
    except Exception:
        print(f"WARNING: invalid response received {response}")
        traceback.print_exc()
        return None

def compile_prompt(work_dir: str, c_code: str, config_file: str, designs: list, info_keys = [
    "C_CODE", 
    "CURR_DESIGN", 
    "OBJECTIVES",
    "PRAGMA_KNOWLEDGE",
    "TARGET_PERFORMANCE", 
    "WARNING_GUIDELINE",
    "OUTPUT_REGULATION",
    # "DESIGN_HINT"
]) -> str:
    prompt_lines = []
    pragma_names = get_default_design(config_file).keys()
    for key in info_keys:
        if key == "C_CODE":
            prompt_lines += [
                "Given the C code with the empty brackets to fill in pragma parameters",
                c_code,
            ]
        elif key == "OBJECTIVES":
            prompt_lines += [
                "Your task is to generate a new pragma variable assignment for the keys: " + ",".join(pragma_names),
            ]
            config_dict = json.load(open(config_file, "r"))["design-space.definition"]
            constraints_str = "\n".join(
                [f"pragma {pragma}'s options are {config_dict[pragma]['options']}" for pragma in config_dict]
            )
            prompt_lines += [
                "Such that the following constraints are honored:\n " + constraints_str
            ]
        elif key == "CURR_DESIGN":
            prompt_lines += [
                "The following are several previous designs with their merlin report and merlin log. ",
                "You should pay attention to the cycle count and the resource utilization in the merlin report and the warning in the merlin log.",
            ]
            for i, design in enumerate(designs):
                prompt_lines += [
                    f"Design {i}:",
                    *[f"pragma {k} = {design[k]}" for k in design],
                    f"The corresponding merlin compilation report for design {i}:", 
                    *extract_perf(os.path.join(work_dir, str(i), "merlin.rpt")),
                    f"The warning when doing merlin compilation for the design {i}:", 
                    *extract_warning(os.path.join(work_dir, str(i), "merlin.log"))
                ]
        elif key == "PRAGMA_KNOWLEDGE":
            prompt_lines += [
                "The following are the pragma knowledge that you can use to generate the new pragma design:", 
                " (1) The pragmas in the C code will be compiled to HLS codes. ",
                " (2) The #pragma ACCEL will affect in the first for loop under it.",
                " (3) The #pragma ACCEL will be followed by three techniques: pipeline, parallel and tile. You want to choose the best combination of these three techniques to optimize the performance according to your anticipation of the cycle count and the resource utilization.",
                "       (3a) For #pragma ACCEL PIPELINE, \"flatten\" will unroll all the for loops below the for loop under this pragma, \"off\" will not apply any pipelining, please only choose between these two options. You might want to choose \"flatten\" if the loop bound is small and the loop body is simple and the DSP utilization is below 0.8 (80%). Otherwise, for the outer loop of a nested loops you might want to choose \"off\".",
                "       (3b) For #pragma ACCEL TILE, it would be better to chose a integer that could divide the corresponding loop bound. Please set it to be 1 if the BRAM utilization is below 0.8 (80%). If the previous point is above 0.8, please choose a integer that could divide the corresponding loop bound and make sure the TILE FACTOR times the PARALLEL FACTOR is smaller than the tripcount.",
                "       (3c) For #pragma ACCEL PARALLEL, it will parallelize the for loop under it. Please choose a integer that could divide the corresponding loop bound. Increasing parallel factor will increase the resource utilization but improve the performance and decease the number of cycles (which is your target)."
            ]
        elif key == "TARGET_PERFORMANCE":
            prompt_lines += [
                "The target cycle of this kernel should be less than 40000. If the cycle count is greater than 10000, please consider using the PIPELINE flatten or increase the PARALLEL FACTOR.",
                "The utilization of DSP, BRAM, LUT, FF and URAM should be as large as possible, but don't exceed 0.8.", 
                "Additionally, the compile time of the merlin should not exceed 40 minutes, meaning that your optimization should not be too aggressive."
            ]
        elif key == "DESIGN_HINT":
            prompt_lines += [
                "First try increasing the parallel factor in the inner most loop."
            ]
        elif key == "WARNING_GUIDELINE":
            prompt_lines += [
                "When you receive the WARNING including tiling factor >= loop tripcount, please decrease the corresponding TILE FACTOR to make sure the TILE FACTOR times the PARALLEL FACTOR is smaller than the tripcount.", 
                "When you receive the WARNING including Coarse-grained pipelining NOT applied on loop, please double check that the PIPELINE FACTOR is either off or flatten.",
            ]
        elif key == "OUTPUT_REGULATION":
            prompt_lines += [
                "Please output the new pragma design as a JSON string. i.e., can be represented as {\"pragma1\": value1, \"pragma2\": value2, ...}"
            ]
        else:
            raise ValueError(f"Invalid key {key}")
    print("\n".join(prompt_lines))
    return "\n".join(prompt_lines)


PRAGMA_KNOWLEDGE = {
    'parallel': [
        f"  (1) Parallel pragram will parallelize the first for loop in the c code under __PARA__.",
        f"  (2) Increasing the parallel factor will increase the resource utilization but improve the performance and decease the number of cycles (which is one of your target).",
        f"  (3) Increasing parallel factor roughly linearly increase the resource utilization within the loop it applies on, so you may scale the factor with respect to the ratio between current utilization with the 80% budget.",
        f"  (4) Increasing the parallel factor will also increase the compilation time, you must decrease the parallel factor if you received the compilation timeout.",
    ], 
    'tile': [
        f"  (1) Tile pragma will tile the first for loop in the c code under __TILE__.",
        f"  (2) Increasing the tile factor will reduce the memory transfer cycles because it will restrict the memory transfer.",
    ],
    'pipeline': [
        f"  (1) Pipeline pragma will affect MULTIPLE loops under __PIPELINE__.",
        f"  (2) The flatten option will unroll all the for loops (which means putting __PARA__ equals to the loop bound in the for loop) under this pragma.",
        f"  (3) Turning off the pipeline will not apply any pipelining, which is useful when you get compilation timeout in the report.",
        f"  (4) Choosing the empty string means coars-grained pipelining, which is useful when you believe the loop inside it has fewer loop-carried dependencies.",
    ]
}

def compile_pragma_update_prompt(best_design: dict, merlin_rpt: str, pragma_name: str, c_code: str, all_options: List[str], pragma_type: str) -> str:
    util_dict = '\n'.join(extract_perf(merlin_rpt))
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n``` with some pragma placeholders for high level synthesis (HLS), your task is to update the {pragma_type} pragma {pragma_name}.",
        f"You must choose one and only one value among {all_options} other than {best_design[pragma_name]} that can optimize the performance the most (reduce the cycle count) while keeping the resource utilization under 80%.",
        f"Note that when {pragma_name} is {best_design[pragma_name]} and: ",
        "\n".join([f"the value of {k} is {v}" for k, v in best_design.items() if k != pragma_name]),
        f"The kernel's results after HLS synthesis are:\n {util_dict}",
        f"To better decide the {pragma_type} factor, here are some knowledge about {pragma_type} pragmas:",
        *PRAGMA_KNOWLEDGE[pragma_type],
        f"You must skip the reasoning and only output in JSON format string, i.e., {{{pragma_name}: value}}"
    ])
    

def compile_arbiter_prompt(best_design: dict, merlin_rpt: str, pragma_updates: List[tuple], c_code: str) -> str:
    util_dict = '\n'.join(extract_perf(merlin_rpt))
    objective = "optimize clock cycles the most." if util_dict != "Compilation Timeout." else "reduce the resource utilization and the compilation time."
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n``` with some pragma placeholders for high level synthesis (HLS), your task is to choose one of the following updates that {objective}",
        "\n".join([f"({i}): change {k} from {best_design[k]} to {v}" for i, (k, v) in enumerate(pragma_updates)]),
        f"Note that when:",
        "\n".join([f"the value of {k} is {v}" for k, v in best_design.items()]),
        f"The kernel's results after HLS synthesis are:\n {util_dict}",
        f"To better understand the problem, here are some knowledge about the HLS pragmas you are encountering:",
        f" For the __PARA__ pragma:", *PRAGMA_KNOWLEDGE['parallel'],
        f" For the __TILE__ pragma:", *PRAGMA_KNOWLEDGE['tile'],
        f" For the __PIPE__ pragma:", *PRAGMA_KNOWLEDGE['pipeline'],
        f"To make better decision, here are some information about the preference:",
        f"  (1) You should prioritize optimizing the __PARA__ pragma first, as it affect the performance the most.",
        f"  (2) If you think all the parallel factors are already optimal, you consider pipeline as the secondary choice. When doing so, you must remember that the pipeline pragma will affect MULTIPLE loops. The flatten option will unroll all the for loops under this pragma. Turning off the pipeline will not apply any pipelining, which is useful when you get compilation timeout in the report.",
        f"  (3) If you think all the parallel factors are already optimal, and the pipeline pragma is already optimal, you can consider the tile pragma. The tile pragma will tile the first for loop in the c code under __TILE__.",
        f"  (4) By default, setting __TILE__ to 1 is perferable.",
        f"  (5) By default, setting __PIPE__ to 1 is perferable.",
        f"Make the update to the current design and output only the new pragma design for the keys: " + ",".join(best_design.keys()) + "as a JSON string. i.e., can be represented as {\"pragma1\": value1, \"pragma2\": value2, ...}",
    ])
    

def parse_merlin_rpt(merlin_rpt: str) -> dict:
    try:
        lines = open(merlin_rpt, "r").readlines()
        target_line_idx = [i for i, l in enumerate(lines) if "Estimated Frequency" in l]
        util_values = lines[target_line_idx[0]+4].split("|")[2:]
        util_keys = ['cycles', 'lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']
        return {util_keys[i]: util_values[i] for i in range(6)}
    except:
        return {}


def extract_perf(merlin_rpt: str):
    merlin_values: dict = parse_merlin_rpt(merlin_rpt)
    return [f"{k} = {v}" for k, v in merlin_values.items()] if merlin_values is not {} else ["Compilation Timeout."]


def extract_warning(input_file):
    if not os.path.exists(input_file): return ["No warning found in the log file"]
    return [l for l in open(input_file, "r").readlines() if "WARNING" in l]


def _parse_options(pragma_name: str, options: str) -> List[str]:
    option_list = eval(re.search(r'\[([^\[\]]+)\]', options).group(0))
    if "PARA" in pragma_name: return [str(x) for x in option_list if option_list[-1] % x == 0]
    return [str(x) for x in option_list]

def compile_design_space(config_file: str) -> dict:
    config_dict = json.load(open(config_file, "r"))["design-space.definition"]
    return {p: _parse_options(p, config_dict[p]['options']) for p in config_dict}

