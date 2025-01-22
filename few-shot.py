import json
from typing import List
import re
import pandas as pd
from util import *

COMPILE_TIMEOUT_MINUTES = 80

KNOWLEDGE = {
    'general': [
		f"Here is some knowledge about the HLS pragmas you are encountering:",
		f"  (1) The pragmas only affect the next for loop after the pragma.",
		f"  (2) The pragmas are __PARA__LX, __PIPE__LX, and __TILE__LX, where LX is the loop name and X is an integer.",
	],
    'parallel': [
		f"Here is some knowledge about the __PARA__LX pragma:",
		f"  (1) The parallel pragma will parallelize the first for loop in the C code under __PARA__.",
		f"  (2) Increasing the parallel factor will increase the resource utilization but improve the performance and decrease the number of cycles (which is one of your targets).",
		f"  (3) Increasing the parallel factor roughly linearly increases the resource utilization within the loop it applies to, so you may scale the factor with respect to the ratio between current utilization and the 80% budget.",
		f"  (4) Increasing the parallel factor will also increase the compilation time; you must decrease the parallel factor if you receive a compilation timeout.",
		f"  (5) The compilation time is positively proportional to the parallel factor; you must choose the parallel factor such that the compilation time is under {COMPILE_TIMEOUT_MINUTES} minutes.",
	],
	'tile': [
		f"Here is some knowledge about the __TILE__LX pragma:",
		f"  (1) The tile pragma will tile the first for loop in the C code under __TILE__.",
		f"  (2) Increasing the tile factor will reduce the memory transfer cycles because it will restrict the memory transfer.", 
		f"  (3) When reducing the compilation time, you should consider setting the tile factor greater than 1.",
	],
	'pipeline': [
		f"Here is some knowledge about the __PIPE__LX pragma:",
		f"  (1) The pipeline pragma will affect MULTIPLE loops under __PIPE__.",
		f"  (2) The flatten option will unroll all the for loops (which means putting __PARA__ equal to the loop bound in the for loop) under this pragma.",
		f"  (3) Turning off the pipeline will not apply any pipelining, which is useful when you get a compilation timeout in the report.",
		f"  (4) Choosing the empty string means coarse-grained pipelining, which increases (roughly doubles) the resource utilization of the for loop's module but potentially improves the performance (reducing cycle count).",
	],
    'arbitrator': [
		f"To make a better decision, here is some information about the preference:",
		f"  (1) You should prioritize optimizing the __PARA__ pragma first, as it affects the performance the most.",
		f"  (2) If you think all the parallel factors are already optimal, you should consider the pipeline as the secondary choice. When doing so, you must remember that the pipeline pragma will affect MULTIPLE loops. The flatten option will unroll all the for loops under this pragma. Turning off the pipeline will not apply any pipelining, which is useful when you get a compilation timeout in the report.",
		f"  (3) If you think all the parallel factors are already optimal, and the pipeline pragma is already optimal, you can consider the tile pragma. The tile pragma will tile the first for loop in the C code under __TILE__.",
		f"  (4) By default, setting __TILE__ to 1 is preferable.",
		f"  (5) By default, setting __PIPE__ to off is preferable.",
	],
}

ROLE = "You are an expert in Merlin pragma insertion."

TASK = {
    "zero-shot": [
        f"For a given c code with pragma, the compiler will generate the number of cycles, the resource utilization, and the compilation time.",
        f"Your task is to select the best pragma parameters from the design space that could minimize the number of cycles, control the resource utilization under 80% and keep the compilation time under 80 mins.",
        f"The priority goal is to minimize the number of cycles.",
    ],
    "one-shot": [
        f"A default design with a compilation result is provided, please make your decision based on the compilation result.",
    ]
}

OUTPUT_FORMAT = "Output only the new design as a JSON string, i.e., can be represented as ```json{\"<pragma1>\": value1, \"<pragma2>\": value2, ...}```, skip the reasoning process."

class CompactListEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, dict):
            formatted_items = []
            for key, value in obj.items():
                if isinstance(value, list):
                    # Compact representation for lists
                    value_str = "[" + ", ".join(json.dumps(item) for item in value) + "]"
                else:
                    # Use default serialization for non-list values
                    value_str = json.dumps(value)
                formatted_items.append(f'"{key}": {value_str}')
            return "{\n  " + ",\n  ".join(formatted_items) + "\n}"
        return super().encode(obj)

def get_batch_openai_response(prompt, model=GPT_MODEL) -> List[str]:
    messages = [
            {"role": "system", "content": "You are an expert in design HLS codes."},
            {"role": "user", "content": prompt}
        ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=10000,  # Set the largest token numbers
        temperature=0.7,  # Control the randomness of the generative result
        n=8,
    )
    # open(OPENAI_LOGFILE, "a").write("\n" + "=" * 80 + "\n" + prompt + "\n" + "-" * 80 + "\n" + response.choices[0].message.content))
    input_tokens = num_tokens_from_messages(messages)
    print(f"input tokens from tiktoken: {input_tokens}")
    print(f"input tokens from oepnai: {response.usage.prompt_tokens}")
    print(f"output tokens from openai: {response.usage.completion_tokens}")
    print(f"total tokens from openai: {response.usage.total_tokens}")
    return [response.choices[i].message.content for i in range(8)]
        
def few_shot_prompt(ds_num:str, benchmark:str, know:bool, arbitrator:bool):
    data_path = "/home/alicewu/lad25/cs259-llm-dse/data"
    C_CODE_FILE = f"{data_path}/lad25/{benchmark}.c"
    CONFIG_FILE = f"{data_path}/lad25/{benchmark}.json"
    c_code = open(C_CODE_FILE, "r").read()
    ds_config = compile_design_space(CONFIG_FILE)
    DATABASE_FILE = f"{data_path}/compilation_results/{benchmark}.csv"
    df = pd.read_csv(DATABASE_FILE)
    default = df.iloc[0].to_dict()
    result_dict = {}
    design_dict = {}
    for key in default:
        if key in ds_config:
            design_dict[key] = default[key]
        else:
            result_dict[key] = default[key]
    prompt = "\n".join([
        f"{ROLE}",
        f"For the given C code\n ```c++ \n{c_code}\n```\n with some pragma placeholders for high-level synthesis (HLS)",
        f"The design space is:\n{json.dumps(ds_config, cls=CompactListEncoder)}",
        *TASK["zero-shot"],
        *(TASK["one-shot"] if ds_num == "one" else []),
        *([f"Given the default design: {json.dumps(design_dict, indent=2)}"] if ds_num == "one" else []),
        *([f"The compilation results are: {json.dumps(result_dict, indent=2)}"] if ds_num == "one" else []),
        *(KNOWLEDGE['general'] if know else []),
        *(KNOWLEDGE['parallel'] if know else []),
        *(KNOWLEDGE['tile'] if know else []),
        *(KNOWLEDGE['pipeline'] if know else []),
        *(KNOWLEDGE['arbitrator'] if arbitrator else []),
        f"{OUTPUT_FORMAT}",
    ])
    PROMPT_FILE = f"{data_path}/few-shot/{benchmark}-{ds_num}-{know}-{arbitrator}.txt"
    open(PROMPT_FILE, "w").write(prompt)
    return prompt

def test():
    # benchmarks = ["atax-medium", "covariance", "fdtd-2d", "gemm-p", "gemver-medium", "jacobi-2d", "symm-opt", "syr2k", "trmm-opt"]
    # shots = ["zero", "one"]
    benchmarks = ["3mm"]
    shots = ["zero"]
    for benchmark in benchmarks:
        for shot in shots:
            output_dir = f"/home/alicewu/lad25/cs259-llm-dse/data/few-shot"
            for know in [False, True]:
                if know:
                    for arbitrator in [False, True]:
                        prompt = few_shot_prompt(shot, benchmark, know, arbitrator)
                        responses = get_batch_openai_response(prompt)
                        print(responses)
                        for i, response in enumerate(responses):
                            design = retrieve_dict_from_response(response)
                            file = f"{output_dir}/{benchmark}-{shot}-{i}-{know}-{arbitrator}.json"
                            open(file, "w").write(json.dumps(design))
                else:
                    prompt = few_shot_prompt(shot, benchmark, know, False)
                    responses = get_batch_openai_response(prompt)
                    print(responses)
                    for i, response in enumerate(responses):
                        design = retrieve_dict_from_response(response)
                        file = f"{output_dir}/{benchmark}-{shot}-{i}-{know}-{False}.json"
                        open(file, "w").write(json.dumps(design))
            
test()
