from config import *
from util import *
import pandas as pd

class Explorer():
    def __init__(self, c_code: str):
        self.c_code = c_code
        self.ds_config = compile_design_space(CONFIG_FILE)
        self.designs = []
        self.prune_space = {}
        self.datas = []
    def explore(self):
        raise NotImplementedError
    def record(self, i_step: int, merlin_rpt: str, design: dict):
        results = parse_merlin_rpt(merlin_rpt)
        results.update(design)
        results["step"] = i_step
        self.datas.append(results)
        pd.DataFrame(self.datas).to_csv(WORK_DIR+"/results.csv", index=False)
    
class Idea0Explorer(Explorer):
    def explore(self):
        prompt_str = compile_prompt(WORK_DIR, self.c_code, CONFIG_FILE, self.designs)
        extra_prompt = ""
        while True:
            response = get_openai_response(prompt_str + extra_prompt)
            curr_design = retrieve_design_from_response(response)
            extra_prompt = input("Waiting for human reponse \n\n\n")
            if extra_prompt == "":
                break
        return curr_design
      
class Idea1Explorer(Explorer):
    def explore(self, i_step: int):
        best_design = self.designs[-1]
        pragma_updates = []
        merlin_rpt = os.path.join(WORK_DIR, str(i_step), "merlin.rpt")
        self.record(i_step, merlin_rpt, best_design)
        for pragma_name in best_design.keys():
            pragma_type = "parallel" if "PARA" in pragma_name else "tile" if "TILE" in pragma_name else "pipeline"
            update_prompt = compile_pragma_update_prompt(best_design, merlin_rpt, pragma_name, self.c_code, self.ds_config[pragma_name], pragma_type)
            pragma_update = retrieve_design_from_response(get_openai_response(update_prompt))
            pragma_updates.append((pragma_name, pragma_update.get(pragma_name, None)))
        prompt = compile_arbiter_prompt(best_design, merlin_rpt, pragma_updates, self.c_code)
        return retrieve_design_from_response(get_openai_response(prompt))
