import argparse
import re, json
from typing import List
import pydot
import pandas as pd

parser = argparse.ArgumentParser(description='Analyze log file')
parser.add_argument('--benchmark', type=str, help='Benchmark name', default="jacobi-2d")
parser.add_argument('--folder', type=str, help='Folder name', default="./exp_opt_arb_bes_2")
args = parser.parse_args()

INT_MAX = 2**31 - 1

def extract_parathesis(s):
    return int(re.search(r'\((.*?)\)', s).group(1).replace("~", "").replace("%", ""))/100 if isinstance(s, str) and "(" in s else INT_MAX
def exclude_parathesis(s):
    return int(s.split("(")[0].strip()) if isinstance(s, str) and "(" in s else INT_MAX

def is_timeout(results: dict) -> bool:
    return results == {} or "cycles" not in results or results["cycles"] == ""

def is_valid(results: dict) -> bool:
    return max([extract_parathesis(results[m]) for m in ['lut utilization', 'FF utilization', 'BRAM utilization' ,'DSP utilization' ,'URAM utilization']]) <= 0.8

def get_perf(results: dict) -> float:
    if is_timeout(results) or not is_valid(results): return float("inf")
    return exclude_parathesis(results["cycles"])

def retrieve_list_from_response(response: str) -> List[dict]:
    return [json.loads(match) for match in re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)]

def retrieve_indices_from_response(response: str) -> List[int]:
    return [int(x) for x in response.strip().split(",")]

def get_type(prompt: str, response: str):
    match_opt = re.search(r"your task is to update the (.*) pragma (.*).", prompt)
    match_arb = re.search(r"your task is to choose (\d+) single updates from the following updates", prompt)
    if match_opt:
        pragma_type, pragma_name = match_opt.groups()
        return "optimizer", {
            "type": pragma_type,
            "pragma_name": pragma_name,
            "updates": retrieve_list_from_response(response),
        }
    if match_arb:
        match_all_options = re.findall(r"change (.*) from (.*) to (.*) while (.*?)\n", prompt)
        num_updates = int(match_arb.group(1))
        indices = retrieve_indices_from_response(response)
        all_options = []
        for option in match_all_options:
            pragma_name, from_val, to_val, design = option
            all_options.append({
                "pragma_name": pragma_name,
                "from_val": from_val,
                "to_val": to_val,
                "design": design,
            })
        return "arbiter", {
            "num_udpate": num_updates,
            "options": all_options,
            "chosen_updates": indices,
        }
    return None, {}

def parse_prompt_response(text):
    prompt_response = [line.split("-"*80) for line in text.split("="*80) if "-"*80 in line]
    return [get_type(*res) for res in prompt_response]

def parse_time_log(log_file: str):
    datas = {}
    for line in open(log_file).readlines():
        match_time = re.search(r'Iteration (\d+), Total runtime: (\d+\.\d+), Iteration runtime: (\d+\.\d+)', line)
        if match_time:
            datas[int(match_time.group(1))] = {
                "total_runtime": float(match_time.group(2)),
                "iteration_runtime": float(match_time.group(3)),
            }
    return datas

pragma_pattern = re.compile(r'__(PARA|TILE|PIPE)__')
pattern = re.compile(r'compilation time|cycles|lut utilization|FF utilization|BRAM utilization|DSP utilization|URAM utilization|__(PARA|TILE|PIPE)__')

def to_dict(data: str):
    return {k.strip(): v.strip() for k, v in [x.split("=") for x in data.split(",")]}

class Analyzer:
    def __init__(self, benchmark: str, folder: str):
        csv_file = f"{folder}/{benchmark}.csv"
        log_file = f"{folder}/{benchmark}.txt"
        time_log = f"{folder}/{benchmark}.log"
        self.logs = parse_prompt_response(open(log_file).read())
        self.times = parse_time_log(time_log)
        self.results = pd.read_csv(csv_file)
        self.node_labels = set()
        self.pragma_names = [col for col in self.results.columns if pragma_pattern.search(col)]

    def serialize_design(self, design: dict, keys: List[str]):
        return '\n'.join([f"{key}={design[key]}" for key in keys])

    def result_at(self, step: int):
        return self.results.iloc[step]

    def design_at(self, step: int):
        return {k: self.result_at(step)[k] for k in self.pragma_names}

    def simulate(self, timeout: int = 8 * 3600):
        curr_step: int = 0
        self.graph = pydot.Dot(graph_type='digraph')
        
        max_iter = 0
        for iter, time_info in self.times.items():
            if time_info["total_runtime"] > timeout: break
            max_iter = iter
        
        for row in self.results.iterrows():
            design = self.design_at(curr_step)
            self.graph.add_node(pydot.Node(self.serialize_design(design, self.pragma_names), shape='box'))
            self.node_labels.add(self.serialize_design(design, self.pragma_names))
            curr_step += 1
            
        max_step, i_iter = 0, 0
        for agent_type, agent_args in self.logs:
            if agent_type == "arbiter":
                i_iter += 1
                if i_iter > max_iter: break
                for idx in agent_args["chosen_updates"]:
                    max_step += 1
                    option = agent_args["options"][idx]
                    pragma_name, from_val, to_val, base_design = option["pragma_name"], option["from_val"], option["to_val"], to_dict(option["design"])
                    old_design = {**base_design, pragma_name: str(to_val)}
                    new_design = {**base_design, pragma_name: str(from_val)}
                    
                    self.graph.add_edge(pydot.Edge(self.serialize_design(old_design, self.pragma_names), self.serialize_design(new_design, self.pragma_names)))
    
        best_perf = min(get_perf(self.result_at(i).to_dict()) for i in range(max_step))
        print(f"Best performance: {best_perf} at step {max_step}, iteration {max_iter}")
    
    def to_dot(self, filename: str):
        self.graph.write_dot(filename)
    
if __name__ == "__main__":
    analyzer = Analyzer(args.benchmark, args.folder)
    analyzer.simulate()
    analyzer.to_dot(f"{args.folder}/{args.benchmark}.dot")