import json
from config import *
from util import *

def llm_dse(c_code, config_file):
    logfile_path = WORK_DIR + "/log.txt"
    logfile = open(logfile_path, "a")
    curr_design: dict = get_default_design(config_file)
    # Keep the list of designs
    designs = []
    designs.append(curr_design)
    for i_steps in range(MAX_ITER):
        prompt_str = ""
        print("-"*80 + f"\nStarting iteration {i_steps}")
        print("-"*80 + f"\nStarting iteration {i_steps}", file=logfile)
        # Generate the C code and the Makefile
        curr_dir = apply_design_to_code(WORK_DIR, c_code, curr_design, i_steps)
        # Compile the C code with pragmas
        run_merlin_compile(curr_dir)
        # Build the prompt based on all the designs in the design list
        prompt_str = compile_prompt(WORK_DIR, c_code, CONFIG_FILE, designs)
        print(prompt_str, file=logfile)
        # Get the response from openai
        response = get_openai_response_o1(prompt_str)
        curr_design = retrieve_design_from_response(response)
        print(curr_design, file=logfile)
        assert isinstance(curr_design, dict), f"expecting dict, got {type(curr_design)}"
        # run_merlin_compile(curr_dir)
        designs.append(curr_design)

    logfile.close()

def idea0_main():
    c_code = open(C_CODE_FILE, "r").read()
    llm_dse(c_code, CONFIG_FILE)

if __name__ == "__main__":
    idea0_main()
