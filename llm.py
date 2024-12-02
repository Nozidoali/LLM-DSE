import openai
import os
import pickle
import json
from dotenv import load_dotenv
from pydantic import BaseModel
# Set the API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DEBUG = False

MAKEFILE_STR = """
# Copyright (C) 2019 Falcon Computing Solutions, Inc. - All rights reserved.
#
# Choose target FPGA platform & vendor
VENDOR=XILINX
#DEVICE=xilinx_aws-vu9p-f1-04261818_dynamic_5_0
#DEVICE=xilinx_u250_xdma_201830_2

#DEVICE=xilinx_vcu1525_xdma_201830_1
# Host Code Compilation settings
#HOST_SRC_FILES=./src/digitrec_host.cpp ./src/util.cpp

# Executable names and arguments
EXE=test
ACC_EXE=test_acc
# Testing mode
EXE_ARGS= data

CXX=g++
CXX_INC_DIRS=-I ./ -I $(MACH_COMMON_DIR)
CXX_FLAGS+= $(CXX_INC_DIRS) -Wall -O3 -std=c++11
ifeq ($(VENDOR),XILINX)
CXX_FLAGS +=-lstdc++ -L$(XILINX_SDX)/lib/lnx64.o
endif

CFLAGS=-I $(XILINX_HLS)/include

# Accelerated Kernel settings
KERNEL_NAME=gemm-p
KERNEL_SRC_FILES=./src/gemm-p.c
KERNEL_INC_DIR=$(CXX_INC_DIRS)

# MerlinCC Options
CMP_OPT=-d11 --attribute burst_total_size_threshold=36700160 --attribute burst_single_size_threshold=36700160 -funsafe-math-optimizations
LNK_OPT=-d11

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

MCC_COMMON_DIR=$(ROOT_DIR)/../mcc_common
include $(MCC_COMMON_DIR)/mcc_common.mk
"""

def read_c_code_from_file(filename):
    # TODO: decrease the number of tokens in the c code
    """
    Reads the content of a C code file and returns it as a string.

    Args:
        filenmae (str): The path to the C code file

    Returns:
        str: The content of the C code file.
    """

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            c_code = file.read()
        return c_code
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        raise

def read_design_point_from_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            designs = pickle.load(file)
        return designs
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        raise
    except pickle.UnpicklingError as e:
        print(f"Error unpickling file '{filename}': {e}")

def read_design_point_from_json(filename):
    with open(filename, 'rb') as file:
        designs = json.load(file)
    return designs

def read_json_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


def extract_perf(input_file):
    perf_data = []
    try:
        with open(input_file, "r") as file:
            lines = file.readlines()
    except Exception as e:
        print(e)
        return ["Compilation Timeout. Please try another design."]

    flag = True
    t = 0

    if DEBUG:
        lines = [
            "Estimated Frequency", 
            "+------------------------+---------------+------------+------------+-------+----------+-------+------+",
            "|         Kernel         |    Cycles     |    LUT     |     FF     | BRAM  |   DSP    | URAM  |Detail|",
            "+------------------------+---------------+------------+------------+-------+----------+-------+------+",
            "|kernel_gemm (gemm-p.c:3)|11047 (0.047ms)|251824 (21%)|372763 (15%)|44 (1%)|1556 (22%)|0 (~0%)|-     |",
            "+------------------------+---------------+------------+------------+-------+----------+-------+------+",
        ]

    for line in lines:
        if "Estimated Frequency" in line:
            flag = True
            continue
        if flag == True:
            perf_data.append(line.strip())
            t += 1
        if t == 5:
            break
    
    try:
        util_keys = ['cycles', 'lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']
        values = perf_data[3].split("|")[2:]
        print(values)
        util_results = {util_keys[i] : values[i] for i in range(6)}
        print(util_results)
        perf_data = []
        for util_key, value in util_results.items():
            perf_data.append(f"{util_key} = {value}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        perf_data = []

    return perf_data

def extract_warning(input_file):
    warning_data = []
    try:
        with open(input_file, "r") as file:
            lines = file.readlines()
    except Exception as e:
        print(e)
        return []
    for line in lines:
        if "WARNING" in line:
            warning_data.append(line.strip())
    return warning_data


def construct_prompt_idea_0(c_code, config, designs, root_dir):
    prompt_lines = [
        "Given the C code with the empty brackets to fill in pragma parameters",
        c_code,
    ]

    prompt_lines.append(
        "The following are several pragma designs with their merlin report and merlin log. "
    )

    for i, design in enumerate(designs):
        keys = []
        for _key, value in design.items():
            keys.append(_key)
            prompt_lines.append(f"pragma {_key} = {value}")

        merlin_rpt_file = root_dir + "/" + str(i) + "/merlin.rpt"
        merlin_log_file = root_dir + "/" + str(i) + "/merlin.log"

        merlin_rpt = extract_perf(merlin_rpt_file)
        prompt_lines.append(
            "The merlin report:"
        )

        prompt_lines.extend(merlin_rpt)

        merlin_log = extract_warning(merlin_log_file)
        if len(merlin_log) != 0:
            prompt_lines.append(
                "The warning when doing merlin compilation:"
            )

            prompt_lines.extend(merlin_log)

    prompt_lines.append(
        "Please generate a new pragma variable assignment for the keys: " + ",".join(keys) 
         + "The prioritized goal is to minimized the number of cycles, the second goal is to eliminate the warnings." 
         + "Only output the JSON file."
    )

    config_dict = json.load(open(config, "r"))
    constraints = [(pragma, config_dict["design-space.definition"][pragma]["options"]) for pragma in config_dict["design-space.definition"]]
    constraints_str = ""
    for pragma, options_str in constraints:
        constraints_str += f"pragma {pragma}'s options are {options_str}\n"

    prompt_lines.append(
        "Such that the following constraints are honored:\n " + constraints_str
    )

    return "\n".join(prompt_lines)

information_list = [
    "The pragmas in the C code will be compiled to HLS codes. ",
    "The #pragma ACCEL will affect in the first for loop under it.",
    "Please notice the #pragma ACCEL pipeline flatten will unroll all the for loops below the for loop under this pragma.",
    "When chosing the parameter for parallel and tile, it would be better to chose a integer that could divide the corresponding loop bound. Additionally, it would be better that the multiplication of the parallel and the tile factor could divide the corresponding loop bound.",
    "The target cycle should be less than 10000.",
    "The utilization of DSP, BRAM, LUT, FF and URAM should be as large as possible, but don't exceed 0.8.",
    "When you receive the WARNING include tiling factor >= loop tripcount, please decrease the corresponding TILE FACTOR."
]

information = " ".join(information_list)


import os
import subprocess
def llm_dse(c_code, config_file):
    config_dict = json.load(open(config_file, "r"))
    design_init: dict = {pragma: config_dict["design-space.definition"][pragma]["default"] for pragma in config_dict["design-space.definition"]}
    root_dir = "./idea0"
    max_steps: int = 20
    i_steps: int = 0
    curr_design = design_init
    designs = []
    designs.append(curr_design)
    flag = False
    while i_steps <= max_steps:
        prompt_str = ""
        if flag:
            print(f"Starting iteration {i_steps}")
            curr_dir = root_dir + f"/{i_steps}/"
            if not os.path.exists(curr_dir):
                os.mkdir(curr_dir)

            curr_src_dir = curr_dir + "src/"
            if not os.path.exists(curr_src_dir):
                os.mkdir(curr_src_dir)
            c_path = curr_src_dir + "gemm-p.c"
            curr_code: str = c_code
            for key, value in curr_design.items():
                curr_code = curr_code.replace("auto{" + key + "}", str(value))
            open(c_path, 'w').write(curr_code)
            open(curr_dir + "Makefile", 'w').write(MAKEFILE_STR)

            # merlin_rpt_file = curr_dir + "merlin.rpt"
            # merlin_log_file = curr_dir + "merlin.log"

            # for debugging:
            if DEBUG:
                subprocess.run(f"echo hi > {merlin_rpt_file}", shell=True)
            else:
                subprocess.run(f"cd {curr_dir} && make mcc_estimate", shell=True, timeout=40*60)
        else:
            print(f"Regenerating iteration {i_steps}")
            prompt_str += "Please avoid generating same design to the given ones.\n"

        prompt_str += construct_prompt_idea_0(c_code, config_file, designs, root_dir)


        prompt_str += information
        print(prompt_str)
        _curr_design = get_openai_response_idea0(prompt_str)

        if isinstance(_curr_design, str):
            success: bool = False
            while not success:
                try:
                    _curr_design = _curr_design.replace("```json", "").replace("```", "").replace("\n", " ").strip()
                    print(_curr_design)
                    import re
                    matches = re.findall(r'\{(.*?)\}', _curr_design)
                    print(matches)
                    assert len(matches) == 1
                    _curr_design = json.loads("{"+matches[0]+"}")
                    curr_design = _curr_design
                    success = True
                    print(curr_design)
                except Exception as e:
                    print(e)
                    print(f"WARNING: invalid response received {_curr_design}")
                    prompt_str = "invalid response format, please generate a response that can be parsed by json.loads"
                    _curr_design = get_openai_response_idea0(prompt_str)
                    exit(0)

        assert isinstance(curr_design, dict), f"expecting dict, got {type(curr_design)}"

        if curr_design not in designs:
            i_steps += 1
            designs.append(curr_design)
            flag = True
        else:
            flag = False

DesignPoint = Dict[str, Union[int, str]]

def gen_key_from_design_point(point: DesignPoint) -> str:
    """Generate a unique key from the given design point.

    Args:
        point: The given design point.

    Returns:
        The generated key in the format of "param1-value1.param2-value2".
    """

    return '.'.join([
        '{0}-{1}'.format(pid,
                         str(point[pid]) if point[pid] else 'NA') for pid in sorted(point.keys())
    ])


def construct_prompt(c_code, designs, max_designs=5, top_k=10):
    """
    Constructs a prompt for the OpenAI API using C code and design points.

    Args:
        c_code (str): The C code with parameter placeholders.
        designs (dict): A dictionary of design points and their performance.
        max_designs (int): Maximum number of design entries to include in the prompt.

    Returns:
        str: The constructed prompt.
    """
    prompt_lines = [
        "Below is the C code with parameter placeholders in {}:",
        c_code,
        "Below are the existing designs:"
    ]
    
    # Include a limited number of design points to avoid token overflow
    # for i, (key, design) in enumerate(list(designs.items())[:max_designs]):
    #     prompt_lines.append(f"\nDesign {i}:")
    #     prompt_lines.append(f"{design.point}")

    for i, design in enumerate(designs[:max_designs]):
        prompt_lines.append(f"Design {i}:{design}")

    prompt_lines.append(
        f"Please output the top-{top_k} design's index to minimize the cycles."
    )
    
    return "\n".join(prompt_lines)

class top_k_response(BaseModel):
    index: list[int]

from typing import Dict, Union
class design_response(BaseModel):
    design: Dict[str, Union[str,int]]


def get_openai_response_idea0(prompt, model="gpt-4o"):
# try:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in design HLS codes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Set the largest token numbers
        temperature=0.7,  # Control the randomness of the generative result
        # response_format=design_response,
    )
    # print(response.choices[0].message)
    chosen_indices = response.choices[0].message.content
    return(chosen_indices)

# Call GPT model
def get_openai_response(prompt, model="gpt-4o"):
# try:
    response = openai.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in design HLS codes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Set the largest token numbers
        temperature=0.7,  # Control the randomness of the generative result
        # response_format=top_k_response,
    )
    # print(response.choices[0].message)
    chosen_indices = response.choices[0].message.parsed
    return(chosen_indices.index)
# except Exception as e:
    # print(f"Error: {e}")
    # return f"Error: {e}"


def run_idea2():
    c_code = read_c_code_from_file("../dse_database/poly/sources/gemm-p_kernel.c")
    # designs = read_design_point_from_pickle("./logs/dse_results_v21_2024-11-30T19-53-37.275177/gemm-p.pickle")
    designs = read_design_point_from_json("gemm-p.json")
    prompt = construct_prompt(c_code, designs, max_designs=100, top_k = 10)
    prompt += information[0]
    prompt += information[1]
    print(prompt)
    indices = get_openai_response(prompt)

    cycles = [11107, 11653, 11047, 12007, 12487, 10927, 11287, 13327, 11647, 10807, 11167, 10651, 10011, 9755, 9947, 9563, 9179, 9563, 10011, 9435, 9435, 9051, 9435, 10267, 9691, 9691, 9691, 9307, 9691, 10139, 10139, 10139, 9563, 9563, 9563, [], [], 9563, 10651, 9755, 9947, 9563, 9435, [], 9435, 10267, 9691, 9691, 9307, 9691, 10139, 10139, 9563, 9563, 9563, 9179, 9179, 9563, 10839, 9585, 9189, 9585, 9849, 9453, 9849, 10311, 9717, 9717, 9717, 9321, 9321, 9717, 9585, 9189, 9585, 9849, 9453, 9849, 9717, 9717, 9321, 9717, 9778, 9370, 9778, 9642, 10050, 9914, 9914, 9506, 9914, 9778, 9778, 10050, 9642, 10050, 9914, 9914, 9506, 9914]
    # cycles = [13270, 298070, 689515, 9099]
    dir ="../res/llm-gemm-p/"
    with open(dir+"child-llm.txt", "w") as file: 
        for index in indices:  
            file.write(f"Design {index}: {cycles[index]} cycles\n")

def run_idea0():
    c_code = read_c_code_from_file("./idea0/gemm-p_kernel.c")
    config_file = "./idea0/gemm-p_ds_config.json"

    llm_dse(c_code, config_file)

if __name__ == "__main__":
    run_idea0()