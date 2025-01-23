import argparse
import re, json
from typing import List

parser = argparse.ArgumentParser(description='Analyze log file')
parser.add_argument('log_file', type=str, help='log file to analyze')
args = parser.parse_args()

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
            design = design.split(" ")[0]
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
    print(len(prompt_response))
    
    curr_iter, curr_step = 0, 0
    for pair in prompt_response:
        prompt, response = pair
        agent_type, agent_args = get_type(prompt, response)
        if agent_type is not None:
            print(f"Agent type: {agent_type}, args: {agent_args}")
        else:
            print(f"Error: cannot decide the type of agent from prompt:\n{prompt}")
            exit()
    
if __name__ == "__main__":
    text = open(args.log_file).read()
    parse_prompt_response(text)