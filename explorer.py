from config import *
from util import *
import pandas as pd

class Explorer():
    def __init__(self, c_code: str, pragma_names: List[str]):
        self.c_code = c_code[:]
        self.ds_config = compile_design_space(CONFIG_FILE)
        self.exploration_history, self.datas = [], []
        self.pragma_names: List[str] = list(pragma_names)[:]
        
    def record_history(self, i_step: int, prev_design: dict, design: dict, hls_results: Dict[str, str], hls_warnings: List[str]):
        pragma_warnings = {}
        if hls_warnings:
            warning_analysis_prompt = compile_warning_analysis_prompt(hls_warnings, self.pragma_names)
            pragma_warnings = retrieve_dict_from_response(get_openai_response(warning_analysis_prompt))
        self.exploration_history.append([i_step, design, hls_results, pragma_warnings])
        self.datas.append({**hls_results, **design, "step": i_step})
        pd.DataFrame(self.datas).to_csv(WORK_DIR+"/results.csv", index=False)        
    
    def load_history(self, design: dict, pragma_name: str) -> Dict[str, str]:
        return {x[1][pragma_name]: format_results(x[2]) for x in self.exploration_history if designs_are_adjacent(x[1], design) and x[1][pragma_name] != design[pragma_name]}
    
    def get_info(self, design: dict) -> int:
        num_total, num_explored = lambda x: len(self.ds_config[x]), lambda x: len(self.load_history(design, x))
        return {
            "remaining space": sum((num_total(pragma_name) - num_explored(pragma_name) for pragma_name in self.pragma_names)), 
            "total space": sum((num_total(pragma_name) for pragma_name in self.pragma_names))
        }
            
    def select_best_designs(self, pragma_name: str) -> List[int]:
        pragma_type = get_pragma_type(pragma_name)
        history = sort_history(self.exploration_history)
        best_design = history[0][1]
        candidates = []
        if pragma_type in ["parallel", "pipeline"]:
            candidates.append(history[0] + [self.get_info(best_design)])
            for step, design, hls_results, pragma_warnings in self.exploration_history:
                if is_timeout(hls_results): continue
                if not is_valid(hls_results): continue
                if designs_are_adjacent(design, best_design): continue
                candidates.append((step, design, hls_results, pragma_warnings, self.get_info(design)))
                if len(candidates) >= NUM_BEST_DESIGN_CANIDATES: break
        else:
            for step, design, hls_results, pragma_warnings in self.exploration_history:
                if not is_timeout(hls_results): continue
                candidates.append((step, design, hls_results, pragma_warnings, self.get_info(design)))
                if len(candidates) >= NUM_BEST_DESIGN_CANIDATES: break
            
        if len(candidates) < NUM_BEST_DESIGNS: return list(map(lambda x: x[0], candidates))
        prompt = compile_best_design_prompt(self.c_code, candidates)
        response = get_openai_response(prompt)
        return list(map(lambda x: candidates[x][0], retrieve_indices_from_response(response)))
    
    def explore(self):
        pragma_updates = []
        for pragma_name in self.pragma_names:
            pragma_type = get_pragma_type(pragma_name)
            for best_design_index in self.select_best_designs(pragma_name):
                _, best_design, hls_results, pragma_warnings = self.exploration_history[best_design_index]
                list_of_warnings = pragma_warnings.get(pragma_name, [])
                exploration_history = self.load_history(best_design, pragma_name)
                update_prompt = compile_pragma_update_prompt(best_design, hls_results, pragma_name, self.c_code, self.ds_config[pragma_name], pragma_type, list_of_warnings, exploration_history)
                pragma_updates.extend((best_design, pragma_name, update.get(pragma_name)) for update in retrieve_list_from_response(get_openai_response(update_prompt)))
        prompt = compile_arbitrator_prompt(self.c_code, pragma_updates, self.pragma_names)
        return [(best_design, {**best_design, **chosen_update}) for chosen_update in retrieve_list_from_response(get_openai_response(prompt))]