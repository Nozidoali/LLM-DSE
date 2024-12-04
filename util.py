import os
import json
import re
import subprocess
import openai
import traceback
import pickle
import torch
from typing import List
from config import *
import signal

def get_default_design(ds_config_file: str) -> dict:
    config_dict = json.load(open(ds_config_file, "r"))["design-space.definition"]
    return {p: config_dict[p]["default"] for p in config_dict}

def load_designs_from_pickle(pickle_file: str, n_best: int = 10) -> List[dict]:
    results = [d for _, d in pickle.load(open(pickle_file, "rb")).items()]
    selected = sorted(results, key=lambda x: x.perf, reverse=True)[:n_best]
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
            process = subprocess.Popen(f"cd {make_dir} && make mcc_estimate", shell=True, preexec_fn = os.setsid)
            process.wait(timeout=40*60)
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
    except Exception as e:
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
    "OUTPUT_REGULATION"
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
                "The target cycle of this kernel should be less than 10000. If the cycle count is greater than 10000, please consider using the PIPELINE flatten or increase the PARALLEL FACTOR.",
                "The utilization of DSP, BRAM, LUT, FF and URAM should be as large as possible, but don't exceed 0.8.", 
                "Additionally, the compile time of the merlin should not exceed 40 minutes, meaning that your optimization should not be too aggressive."
            ]
        elif key == "WARNING_GUIDELINE":
            prompt_lines += [
                "When you receive the WARNING including tiling factor >= loop tripcount, please decrease the corresponding TILE FACTOR to make sure the TILE FACTOR times the PARALLEL FACTOR is smaller than the tripcount.", 
                "When you receive the WARNING including Coarse-grained pipelining NOT applied on loop, please double check that the PIPELINE FACTOR is either off or flatten.",
                "When you receive the WARNING including "
            ]
        elif key == "OUTPUT_REGULATION":
            prompt_lines += [
                "Please output the new pragma design as a JSON string. i.e., can be represented as {\"pragma1\": value1, \"pragma2\": value2, ...}"
            ]
        else:
            raise ValueError(f"Invalid key {key}")
    print("\n".join(prompt_lines))
    return "\n".join(prompt_lines)


def extract_perf(input_file):
    lines = open(input_file, "r").readlines()
    target_line_idx = [i for i, l in enumerate(lines) if "Estimated Frequency" in l]
    try:
        util_values = lines[target_line_idx[0]+4].split("|")[2:]
        util_keys = ['cycles', 'lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']
        return [f"{util_keys[i]} = {util_values[i]}" for i in range(6)]
    except:
        print(f"WARNING: cannot extract performance data from {input_file}")
        return [f"WARNING: cannot extract performance data from {input_file}"]

def extract_warning(input_file):
    if not os.path.exists(input_file): return ["No warning found in the log file"]
    return [l for l in open(input_file, "r").readlines() if "WARNING" in l]
