import json
from config import *
from util import *

class Idea0Explorer(Explorer):
    def explore(self):
        prompt_str = compile_prompt(WORK_DIR, self.c_code, CONFIG_FILE, self.designs)
        print(prompt_str, file=self.logfile)
        extra_prompt = ""
        while True:
            response = get_openai_response(prompt_str + extra_prompt)
            curr_design = retrieve_design_from_response(response)
            print(curr_design, file=self.logfile)
            extra_prompt = input("Waiting for human reponse \n\n\n")
            if extra_prompt == "":
                break
        return curr_design
