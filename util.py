import os
import json
import re
import subprocess
import openai
from openai import OpenAI
import traceback
import pickle
from typing import List, Dict, Union, Optional, Tuple
from config import *
import signal
import time
import random
from datetime import timedelta
import tiktoken


def extract_parathesis(s):
    return int(re.search(r'\((.*?)\)', s).group(1).replace("~", "").replace("%", ""))/100 if isinstance(s, str) and "(" in s else float("inf")

def exclude_parathesis(s):
    return int(s.split("(")[0].strip()) if isinstance(s, str) and "(" in s else float("inf")

def get_default_design(ds_config_file: str) -> dict:
    config_dict = json.load(open(ds_config_file, "r"))["design-space.definition"]
    return {p: str(config_dict[p]["default"]) for p in config_dict}

def is_timeout(results: dict) -> bool:
    return results == {} or "cycles" not in results or results["cycles"] == ""

def is_valid(results: dict) -> bool:
    return max([extract_parathesis(results[m]) for m in ['lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']]) <= 0.8

def get_perf(results: dict) -> float:
    if is_timeout(results) or not is_valid(results): return float("inf")
    return exclude_parathesis(results["cycles"])

def format_design(design: dict, exclude: list = None) -> str:
    return ", ".join([f"{k} = {v}" for k, v in design.items() if not exclude or k not in exclude])

def format_results(results: dict) -> str:
    if is_timeout(results): return "Compilation Timeout."
    return ", ".join([f"{k} = {v}" for k, v in results.items()])

def format_list(l: list) -> str:
    return ", ".join([f"\"{x}\"" for x in l])

def format_example(design: dict, results: dict, warnings: List[str], reflection=None) -> str:
    return ", ".join(filter(None, [
        f"**{reflection}**" if reflection else "",
        f"When {format_design(design)}",
        f"the result is: {format_results(results)}",
        (f"with warnings: {format_list(warnings)}" if warnings else ""),
    ]))
    
def extract_dict(design_str: str) -> Dict[str, str]:
    return {k.strip(): v.strip() for k, v in re.findall(r'(\w+(?: \w+)*)\s*=\s*([^\s,]+)', design_str)}

def designs_are_adjacent(design1: dict, design2: dict) -> bool:
    return sum([design1[k] != design2[k] for k in design1.keys()]) == 1

def designs_are_equal(design1: dict, design2: dict) -> bool:
    return all([design1[k] == design2[k] for k in design1.keys()])

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

def _rand_result(rand=False):
    if not rand: return "0 (0%)"
    return f"{random.randint(0, 100)} ({random.randint(0, 100)}%)"

def eval_design(work_dir: str, c_code: str, curr_design: dict, idx: int) -> Tuple[Dict[str, str], List[str]]:
    if DEBUG_MERLIN:
        print(format_design(curr_design)) 
        return {"cycles": _rand_result(), "lut utilization": _rand_result(), "FF utilization": _rand_result(), "BRAM utilization": _rand_result(), "DSP utilization": _rand_result(), "URAM utilization": _rand_result()}, []
    if DATABASE_IS_VALID:
        import pandas as pd
        df = pd.read_csv(DATABASE_FILE)
        for col in list(curr_design.keys()): 
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(int).astype(str)
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

def get_openai_response(prompt, model=GPT_MODEL, temperature=0.7) -> str:
    if model == GPT_MODEL:
        failed = True
        while failed:
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in design HLS codes."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10000,  # Set the largest token numbers
                    temperature=temperature,  # Control the randomness of the generative result
                ).choices[0].message.content if not DEBUG_OPENAI else input(f"{prompt}\n\033[33mENTER response: >\033[0m\n")
                failed = False
            except:
                failed = True
                time.sleep(2)
        open(OPENAI_LOGFILE, "a").write("\n" + "=" * 80 + "\n" + prompt + "\n" + "-" * 80 + "\n" + response)
        return(response)
    elif model == DEEPSEEK_MODEL:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in design HLS codes."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            # response_format={
            #     'type': 'json_object'
            # },
            max_tokens=8192,  # Set the largest token numbers
            temperature=1.3
        ).choices[0].message.content
        open(OPENAI_LOGFILE, "a").write("\n" + "=" * 80 + "\n" + prompt + "\n" + "-" * 80 + "\n" + response)
        return(response)
        
def handle_response_exceptions(func):
    def wrapper(response: str):
        try:
            return func(response)
        except Exception:
            print(f"WARNING: invalid response received {response}")
            traceback.print_exc()
            raise Exception("Invalid response received.")
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

def compile_best_design_prompt(c_code: str, candidates: list) -> str:
    n_best_designs: int = min(NUM_BEST_DESIGNS, len(candidates))
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high-level synthesis (HLS), your task is to choose the top {n_best_designs} best designs among the following options.",
        *[f" {i}: {format_example(design, hls_results, [], reflection)} The remaining search space is {info['remaining space']}."
          for i, (_, design, hls_results, _, reflection, info) in enumerate(candidates)],
        *KNOWLEDGE_DATABASE['best_design'],
        f"Note that you are doing a design space exploration, and you must find the design that can be further optimized.",
        REGULATE_OUTPUT("index list", len(candidates), n_best_designs),
    ])
    
def compile_general_reflection_prompt(c_code: str, prev_design: dict, curr_design: dict, prev_hls_results: Dict[str, str], prev_pragma_warnings: Dict[str, List[str]], curr_hls_results: Dict[str, str], curr_pragma_warnings: Dict[str, List[str]], pragma_names: List[str]) -> str:
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high-level synthesis (HLS), your task is to reflect on the two following designs.", 
        format_example(prev_design, prev_hls_results, prev_pragma_warnings),
        format_example(curr_design, curr_hls_results, curr_pragma_warnings), 
        *KNOWLEDGE_DATABASE['objective'],
        f"Your task is to output a JSON string with the pragma name as the key and the list of reflections as the value.", 
        *KNOWLEDGE_DATABASE['reflection'], 
        f"The list of pragma names is:",
        "\n".join([f"\t{i}. {pragma_name}" for i, pragma_name in enumerate(pragma_names)]),
        f"You must output a JSON string with the pragma name as the key and the list of reflection strings.",
        f"You don't need to cover all the pragmas, only the ones that has REALLY constructive suggestion.", 
        f"You could generate at most {SELF_REFLECTION_LIMIT} reflections for each pragma, and each reflection should be a sentence with at most {SELF_REFLECTION_WORD_LIMIT} words.",
        f"Never output the reasoning and you must make sure the JSON string is valid.",
    ])
    
def compile_pruning_reflection_prompt(prev_design: dict, curr_design: dict, prev_hls_results: Dict[str, str], curr_hls_results: Dict[str, str]) -> str:
    return "\n".join([
        f"You task is to decide for the DSE, whether a new design is useful or useless to be updated in the database."
        f"The previous design is {format_design(prev_design)} with the results {format_results(prev_hls_results)}",
        f"The current design is {format_design(curr_design)} with the results {format_results(curr_hls_results)}",
        *KNOWLEDGE_DATABASE['pruning_reflection'],
        f"Output a comment no longer than {SELF_REFLECTION_WORD_LIMIT} words for the current design.",
        f"The design is"
    ])

def compile_warning_analysis_prompt(warnings: List[str], pragma_names: List[str]) -> str:
    return "\n".join([
        f"You must decide for each pragma below a list of warnings that you should consider when updating the pragma.",
        *KNOWLEDGE_DATABASE['warning_analysis'],
        f"Based on the HLS compilation log, there are some warnings that you should assign to pragmas:",
        "\n".join([f"\t{i}. {warning}" for i, warning in enumerate(warnings)]),
        f"The list of pragmas is:",
        "\n".join([f"\t{i}. {pragma_name}" for i, pragma_name in enumerate(pragma_names)]),
        f"You must output a JSON string with the pragma name as the key and the list of original warnings as the value.",
        f"You don't need to include all the warnings, only the ones that are relevant to a pragma.",
        f"Never output the reasoning and you must make sure the JSON string is valid.",
    ])

def compile_pragma_update_prompt(best_design: dict, hls_results: Dict[str, str], pragma_name: str, c_code: str, all_options: List[str], pragma_type: str, hls_warnings: List[str], explored: Dict[str, str], self_reflection: List[str] = []) -> str:
    n_optimizations: int = min(NUM_OPTIMIZATIONS, len(all_options) - 1) if pragma_type != "pipeline" else 1
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high-level synthesis (HLS), your task is to update the {pragma_type} pragma {pragma_name}.",
        f"You must choose {n_optimizations} values among ({format_list(all_options)})",
        *KNOWLEDGE_DATABASE['objective'],
        format_example(best_design, hls_results, hls_warnings),
        "\n".join([f"when changing {pragma_name} to {k}, the results are: {v}" for k, v in explored.items()]),
        *KNOWLEDGE_DATABASE[pragma_type], 
        (f"Based on the previous experience, you should consider the following reflections:\n" + "\n".join(self_reflection) if self_reflection != [] else ""),
        f"Your response should not contain the reasoning and only output at most {n_optimizations} separate JSON strings, i.e., ```json{{\"{pragma_name}\": <value>}}```, with {n_optimizations} different values."
    ])

def compile_arbitrator_prompt(c_code: str, pragma_updates: List[tuple]) -> str:
    n_designs: int = min(NUM_CHOSENS, len(pragma_updates))
    return "\n".join([
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high-level synthesis (HLS), your task is to choose {n_designs} single updates from the following updates that optimize clock cycles the most.",
        "\n".join([f"({i}): change {k} from {d[k]} to {v} while {format_design(d, exclude=[k])}" 
            for i, (d, k, v) in enumerate(pragma_updates)]),
        *KNOWLEDGE_DATABASE['general'],
        *KNOWLEDGE_DATABASE['parallel'],
        *KNOWLEDGE_DATABASE['pipeline'],
        *KNOWLEDGE_DATABASE['tile'],
        *KNOWLEDGE_DATABASE['arbitrator'],
        REGULATE_OUTPUT("index list", len(pragma_updates), n_designs),
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

def get_pragma_type(pragma_name: str) -> str:
    return "parallel" if "PARA" in pragma_name else "tile" if "TILE" in pragma_name else "pipeline"

def get_loop_name(pragma_name: str) -> str:
    return pragma_name.split('__')[-1]

def num_tokens_from_messages(messages, model="gpt-4o"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
