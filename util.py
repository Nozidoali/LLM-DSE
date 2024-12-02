import os
import json
import re
import subprocess
import openai
from config import *

def get_default_design(ds_config_file: str) -> dict:
    config_dict = json.load(open(ds_config_file, "r"))["design-space.definition"]
    return {p: config_dict[p]["default"] for p in config_dict}

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
    else: subprocess.run(f"cd {make_dir} && make mcc_estimate", shell=True, timeout=40*60)

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

def retrieve_design_from_response(response: str) -> dict:
    try:
        _response = response.replace("```json", "").replace("```", "").replace("\n", " ").strip()
        print(f"INFO: response received {_response}")
        matches = re.findall(r'\{(.*?)\}', _response)
        design = json.loads("{"+matches[0]+"}")
        return design
    except:
        print(f"WARNING: invalid response received {response}")
        return None

def compile_prompt(work_dir: str, c_code: str, config_file: str, designs: list, info_keys = [
    "C_CODE", 
    "CURR_DESIGN", 
    "OBJECTIVES",
    "PRAGMA_KNOWLEDGE",
    "TARGET_PERFORMANCE", 
    "WARNING_GUIDELINE"
]) -> str:
    prompt_lines = []
    pragma_names = get_default_design(config_file).keys()
    for key in info_keys:
        if key == "C_CODE":
            prompt_lines += [
                "Given the C code with the empty brackets to fill in pragma parameters",
                c_code,
            ]
        elif key == "CURR_DESIGN":
            prompt_lines += [
                "The following are several pragma designs with their merlin report and merlin log. "
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
        elif key == "OBJECTIVES":
            prompt_lines.append(
                "Please generate a new pragma variable assignment for the keys: " + ",".join(pragma_names) 
                + "The prioritized goal is to minimized the number of cycles, the second goal is to eliminate the warnings." 
                + "Only output the JSON file."
            )
            config_dict = json.load(open(config_file, "r"))["design-space.definition"]
            constraints_str = "\n".join(
                [f"pragma {pragma}'s options are {config_dict[pragma]['options']}" for pragma in config_dict]
            )
            prompt_lines.append(
                "Such that the following constraints are honored:\n " + constraints_str
            )
        elif key == "PRAGMA_KNOWLEDGE":
            prompt_lines += [
                "The following are the pragma knowledge that you can use to generate the new pragma design:", 
                " (1) The pragmas in the C code will be compiled to HLS codes. ",
                " (2) The #pragma ACCEL will affect in the first for loop under it.",
                " (3) Please notice the #pragma ACCEL pipeline flatten will unroll all the for loops below the for loop under this pragma.",
                " (4) When chosing the parameter for parallel and tile, it would be better to chose a integer that could divide the corresponding loop bound. Additionally, it would be better that the multiplication of the parallel and the tile factor could divide the corresponding loop bound.",
            ]
        elif key == "TARGET_PERFORMANCE":
            prompt_lines += [
                "The target cycle should be less than 10000.",
                "The utilization of DSP, BRAM, LUT, FF and URAM should be as large as possible, but don't exceed 0.8."
            ]
        elif key == "WARNING_GUIDELINE":
            prompt_lines += [
                "When you receive the WARNING include tiling factor >= loop tripcount, please decrease the corresponding TILE FACTOR.",
            ]
        else:
            raise ValueError(f"Invalid key {key}")
    return "\n".join(prompt_lines)


def extract_perf(input_file):
    lines = open(input_file, "r").readlines()
    target_line_idx = [i for i, l in enumerate(lines) if "Target Performance" in l]
    try:
        util_values = lines[target_line_idx[0]+3].split("|")[2:]
        util_keys = ['cycles', 'lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']
        return [f"{util_keys[i]} = {util_values[i]}" for i in range(6)]
    except:
        print(f"Error: cannot extract performance data from {input_file}")
        return [f"Error: cannot extract performance data from {input_file}"]

def extract_warning(input_file):
    return [l for l in open(input_file, "r").readlines() if "WARNING" in l]