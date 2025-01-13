from config import *
from util import *
import pandas as pd

class Explorer():
    def __init__(self, c_code: str, pragma_names: List[str]):
        self.c_code = c_code[:]
        self.ds_config = compile_design_space(CONFIG_FILE)
        self.exploration_history, self.datas = [], []
        self.best_design_history = []
        self.pragma_names: List[str] = list(pragma_names)[:]
        
    def record_history(self, i_step: int, design: dict, hls_results: Dict[str, str], hls_warning: List[str]):
        self.exploration_history.append([i_step, design, hls_results, hls_warning])
        self.datas.append({**hls_results, **design, "step": i_step})
        pd.DataFrame(self.datas).to_csv(WORK_DIR+"/results.csv", index=False)        
    
    def load_history(self, design: dict, pragma_name: str) -> Dict[str, str]:
        return {x[1][pragma_name]: format_results(x[2]) for x in self.exploration_history if designs_are_adjacent(x[1], design) and x[1][pragma_name] != design[pragma_name]}
    
    def design_space_info(self, design: dict) -> int:
        num_total, num_explored = lambda x: len(self.ds_config[x]), lambda x: len(self.load_history(design, x))
        return {
            "remaining space": sum((num_total(pragma_name) - num_explored(pragma_name) for pragma_name in self.pragma_names)), 
            "total space": sum((num_total(pragma_name) for pragma_name in self.pragma_names))
        }
            
    def select_best_designs(self):
        best_design_info = {i: self.design_space_info(design) for i, (_, design, _, _) in enumerate(self.exploration_history)}
        prompt = compile_best_design_prompt(self.c_code, self.exploration_history, self.best_design_history, best_design_info)
        response = get_openai_response(prompt)
        return retrieve_indices_from_response(response)
    
    def explore(self):
        next_designs = []
        for best_design_index in self.select_best_designs():
            self.best_design_history.append(best_design_index)
            _, best_design, hls_results, hls_warnings = self.exploration_history[best_design_index]
            if len(hls_warnings) == 0: pragma_warnings = {}
            else:
                warning_analysis_prompt = compile_warning_analysis_prompt(hls_warnings, best_design.keys())
                pragma_warnings = retrieve_dict_from_response(get_openai_response(warning_analysis_prompt))
            pragma_updates = []
            for pragma_name in best_design.keys():
                list_of_warnings = pragma_warnings.get(pragma_name, [])
                exploration_history = self.load_history(best_design, pragma_name)
                pragma_type = "parallel" if "PARA" in pragma_name else "tile" if "TILE" in pragma_name else "pipeline"
                update_prompt = compile_pragma_update_prompt(best_design, hls_results, pragma_name, self.c_code, self.ds_config[pragma_name], pragma_type, list_of_warnings, exploration_history)
                pragma_updates.extend((pragma_name, update.get(pragma_name)) for update in retrieve_list_from_response(get_openai_response(update_prompt)))
            prompt = compile_arbitrator_prompt(best_design, hls_results, list_of_warnings, pragma_updates, self.c_code)
            next_designs.extend([{**best_design, **chosen_update} for chosen_update in retrieve_list_from_response(get_openai_response(prompt)) if {**best_design, **chosen_update} not in next_designs])
        return next_designs