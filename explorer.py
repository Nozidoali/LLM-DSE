from config import *
from util import *
import pandas as pd

class Explorer():
    def __init__(self, c_code: str):
        self.c_code = rewrite_c_code(c_code) if ENABLE_CODE_ANAL_AGENT else c_code
        self.ds_config = compile_design_space(CONFIG_FILE)
        self.exploration_history, self.datas = [], []
        
    def record_history(self, i_step: int, design: dict, hls_results: Dict[str, str], hls_warning: List[str]):
        self.exploration_history.append([i_step, design, hls_results, hls_warning])
        self.datas.append({"step": i_step, **hls_results, **design})
        pd.DataFrame(self.datas).to_csv(WORK_DIR+"/results.csv", index=False)
    
    def load_history(self, design: dict, pragma_name: str) -> Dict[str, str]:
        return {x[1][pragma_name]: format_results(x[2]) for x in self.exploration_history if designs_are_adjacent(x[1], design) and x[1][pragma_name] != design[pragma_name]}
    
    def load_best_design(self):
        prompt = compile_best_design_prompt(self.c_code, self.exploration_history)
        response = get_openai_response(prompt)
        print(response)
        # index = retrieve_index_from_response(response)
        indices = retrieve_indices_from_response(response)
        return [self.exploration_history[index] for index in indices]
    
    def explore(self):
        _, best_design, hls_results, hls_warning = self.load_best_design()
        warning_analysis_prompt = compile_warning_analysis_prompt(hls_warning, best_design.keys())
        pragma_warnings = retrieve_dict_from_response(get_openai_response(warning_analysis_prompt))
        pragma_updates = []
        for pragma_name in best_design.keys():
            list_of_warnings = pragma_warnings.get(pragma_name, [])
            exploration_history = self.load_history(best_design, pragma_name)
            pragma_type = "parallel" if "PARA" in pragma_name else "tile" if "TILE" in pragma_name else "pipeline"
            update_prompt = compile_pragma_update_prompt(best_design, hls_results, pragma_name, self.c_code, self.ds_config[pragma_name], pragma_type, list_of_warnings, exploration_history)
            pragma_update = retrieve_dict_from_response(get_openai_response(update_prompt))
            pragma_updates.append((pragma_name, pragma_update.get(pragma_name, None)))
        prompt = compile_arbitrator_prompt(best_design, hls_results, pragma_updates, self.c_code)
        return {**best_design, **retrieve_dict_from_response(get_openai_response(prompt))}
