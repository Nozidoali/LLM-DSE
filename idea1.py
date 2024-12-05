from util import *
import re
import os

def pragma_list(ds_config_file):
    pragma_names = get_default_design(ds_config_file).keys()
    return pragma_names


def prune_design_space(last_design, curr_design, i_step, prune_space):
    last_rpt_file = os.path.join(WORK_DIR, str(i_step-1), "merlin.rpt")
    curr_rpt_file = os.path.join(WORK_DIR, str(i_step), "merlin.rpt")
    last_util = extract_perf(last_rpt_file)
    curr_util = extract_perf(curr_rpt_file)
    prune = False
    if curr_util == "Compilation Timeout" and last_util != "Compilation Timeout":
        prune = True
    elif isinstance(curr_util, dict):
        curr_util = {k: int(re.search(r'\d+(?=%)', v).group()) for k, v in curr_util.items()}
        last_util = {k: int(re.search(r'\d+(?=%)', v).group()) for k, v in last_util.items()}
        for k in curr_util:
            if curr_util[k] > 80 and last_util[k] < 80:
                prune = True
    if prune:
        for k, v in curr_design.items():
            if k in last_design and last_design[k] != v:
                if k in prune_space:
                    prune_space[k].append(v)
                else:
                    prune_space[k] = [v]


def para_llm(rpt, pragma_key, pragma_value, c_code, prune_space):
    util_dict = extract_perf(rpt)
    config_dict = json.load(open(CONFIG_FILE, "r"))["design-space.definition"]
    prompt_lines = [
        f"For the given C code {c_code}",
        f"The last design for {pragma_key} is {pragma_value}",
        f"which has the utilization {util_dict} after HLS synthesis.",
        # f"For aggresive choice, the factor you choose divdes by {pragma_value} could be close to."
        f"Parallel pragram will parallelize the first for loop in the c code under {pragma_key}.",
        f"Please choose one factor among {config_dict[pragma_key]['options']} that could divide the corresponding loop bound,"
        f"The chosen factor should not be in: {prune_space.get(pragma_key, [])}",
        f"The chosen factor should try to make use of 80% of the resource utilization."
        f"Only output in JSON format: {{{pragma_key}: value}}"
    ]
    prompt =  "\n".join(prompt_lines)
    print("INFO: prompt for a parallel pragma design point", prompt)
    response = get_openai_response(prompt)
    pragma_update = retrieve_design_from_response(response)
    return pragma_key, pragma_update.get(pragma_key, None)


def tile_llm(pragma_key, pragma_value):
    prompt_lines = [
        f"Given the tile pragma {pragma_key}",
        "Set the tile factor to 1.",
        f"Only output in JSON format: {{{pragma_key}: value}}"
    ]
    prompt = "\n".join(prompt_lines)
    print("INFO: prompt for a tile pragma design point", prompt)
    response = get_openai_response(prompt)
    pragma_update = retrieve_design_from_response(response)
    return pragma_key, pragma_update.get(pragma_key, None)


def log_prompt(input_file, pragma_names, ):
    warning = extract_warning(input_file)


class Idea1Explorer(Explorer):
    def explore(self, i_step):
        if i_step >= 1:
            prune_design_space(self.designs[-2], self.designs[-1], i_step, self.prune_space)
        last_design = self.designs[-1]
        pragma_updates = []
        for pragma_key, pragma_value in last_design.items():
            if "PARA" in pragma_key:
                print(os.path.join(WORK_DIR, str(i_step), "merlin.rpt"))
                pragma_update = para_llm(os.path.join(WORK_DIR, str(i_step), "merlin.rpt"), pragma_key, pragma_value, self.c_code, self.prune_space)
            elif "TILE" in pragma_key:
                pragma_update = tile_llm(pragma_key, pragma_value)
            assert pragma_update is not None and isinstance(pragma_update, tuple), f"Invalid pragma update: {pragma_update}"
            pragma_updates.append(pragma_update)
        objective = "latency"
        prompt_lines = [
            f"For the given C code {self.c_code} and the current design {self.designs[-1]}",
            f"Please choose one of the following updates that optimized {objective} the most.",
            "\n".join([f"({i}): change {k} from {self.designs[-1][k]} to {v}" for i, (k, v) in enumerate(pragma_updates)]),
            f"Make the update to the current design and output only the new pragma design for the keys: " + ",".join(last_design.keys()) + "as a JSON string. i.e., can be represented as {\"pragma1\": value1, \"pragma2\": value2, ...}"
        ]
        prompt = "\n".join(prompt_lines)
        response = get_openai_response(prompt)
        curr_design = retrieve_design_from_response(response)
        return curr_design





