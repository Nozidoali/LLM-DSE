import json
from config import *
from util import *

def llm_dse(c_code, config_file):
    curr_design: dict = get_default_design(config_file)
    designs = [curr_design]
    flag = False
    for i_steps in range(MAX_ITER):
        prompt_str = ""
        if flag:
            print("-"*80 + f"\nStarting iteration {i_steps}")
            curr_dir = apply_design_to_code(WORK_DIR, c_code, curr_design, i_steps)
            run_merlin_compile(curr_dir)
        else:
            print(f"Regenerating iteration {i_steps}")
            prompt_str += "Please avoid generating same design to the given ones.\n"
        prompt_str = compile_prompt(WORK_DIR, c_code, CONFIG_FILE, designs)
        curr_design = retrieve_design_from_response(prompt_str)
        assert isinstance(curr_design, dict), f"expecting dict, got {type(curr_design)}"
        designs.append(curr_design)

def idea0_main():
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code, CONFIG_FILE)

if __name__ == "__main__":
    idea0_main()
