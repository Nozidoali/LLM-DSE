from config import *
from util import *
import pandas as pd

class Explorer():
    def __init__(self, c_code: str, pragma_names: List[str]):
        self.c_code = c_code[:]
        self.ds_config = compile_design_space(CONFIG_FILE)
        self.exploration_history, self.datas = [], []
        self.pragma_names: List[str] = list(pragma_names)[:]
        self.optimizer_reflections: Dict[str, List[str]] = {p: [] for p in pragma_names}
        
    def record_history(self, i_step: int, design: dict, hls_results: Dict[str, str], pragma_warnings: Dict[str, List[str]]):
        self.exploration_history.append([i_step, design, hls_results, pragma_warnings])
        self.datas.append({**hls_results, **design, "step": i_step})
        pd.DataFrame(self.datas).to_csv(WORK_DIR+"/results.csv", index=False)        
    
    def load_history(self, design: dict, pragma_name: str) -> Dict[str, str]:
        return {x[1][pragma_name]: format_results(x[2]) for x in self.exploration_history if designs_are_adjacent(x[1], design) and x[1][pragma_name] != design[pragma_name]}
    
    def load_results(self, design: dict) -> Dict[str, str]:
        for x in self.exploration_history:
            if designs_are_equal(x[1], design): return x[2], x[3]
        return None, None
    
    def get_info(self, design: dict) -> int:
        num_total, num_explored = lambda x: len(self.ds_config[x]), lambda x: len(self.load_history(design, x))
        return {
            "remaining space": sum((num_total(pragma_name) - num_explored(pragma_name) for pragma_name in self.pragma_names)), 
            "total space": sum((num_total(pragma_name) for pragma_name in self.pragma_names))
        }
        
    def analyze_warnings(self, warnings: List[str]) -> Dict[str, List[str]]:
        pragma_warnings = {}
        if AUTO_WARNING_ANALYSIS:
            return {pragma_name: [_w for _w in warnings if get_loop_name(pragma_name) in _w] for pragma_name in self.pragma_names}
        if warnings:
            try:
                warning_analysis_prompt = compile_warning_analysis_prompt(warnings, self.pragma_names)
                pragma_warnings = retrieve_dict_from_response(get_openai_response(warning_analysis_prompt))
            except Exception as e:
                pass
        return pragma_warnings
    
    def filter_history(self, history, get_result: callable):
        seen = set()
        unique_history = []
        for x in history:
            hls_result = get_result(x)
            if is_timeout(hls_result): 
                unique_history.append(x)
                continue
            identifier = tuple(hls_result[k] for k in RESULT_KEYS)
            if identifier not in seen:
                seen.add(identifier)
                unique_history.append(x)
        return unique_history

    def select_best_designs(self, pragma_name: str) -> List[int]:
        pragma_type = get_pragma_type(pragma_name)
        history = sorted(self.exploration_history, key=lambda x: get_perf(x[2]))
        history = self.filter_history(history, lambda x: x[2])
        best_design = history[0][1]
        candidates = []
        if pragma_type in ["parallel", "pipeline"]:
            best_design_info = self.get_info(best_design)
            if best_design_info['remaining space'] != 0:
                candidates.append(history[0] + [self.get_info(best_design)])
            for step, design, hls_results, pragma_warnings in history[1:]:
                if is_timeout(hls_results) or not is_valid(hls_results): continue
                design_info = self.get_info(design)
                if design_info['remaining space'] != 0:
                    candidates.append((step, design, hls_results, pragma_warnings, self.get_info(design)))
                if len(candidates) >= NUM_BEST_DESIGN_CANIDATES: break
        else:
            for step, design, hls_results, pragma_warnings in history:
                if is_timeout(hls_results) or not is_valid(hls_results):
                    candidates.append((step, design, hls_results, pragma_warnings, self.get_info(design)))
                if len(candidates) >= NUM_BEST_DESIGN_CANIDATES: break
            random.shuffle(candidates)
        if len(candidates) < NUM_BEST_DESIGNS: return list(map(lambda x: x[0], candidates))
        if AUTO_BEST_DESIGN: return list(map(lambda x: x[0], candidates[:NUM_BEST_DESIGNS]))
        try:
            prompt = compile_best_design_prompt(self.c_code, candidates)
            response = get_openai_response(prompt)
            return list(map(lambda x: candidates[x][0], retrieve_indices_from_response(response)))
        except Exception as e:
            return list(map(lambda x: x[0], candidates[:NUM_BEST_DESIGNS]))
    
    def propose_update(self, from_idx: int, pragma_name: str) -> dict:
        pragma_type = get_pragma_type(pragma_name)
        _, best_design, hls_results, warnings = self.exploration_history[from_idx]
        explored_values = self.load_history(best_design, pragma_name)
        filtered_explored_values = self.filter_history(list(explored_values.values()), extract_dict)
        explored_hls_results = list(explored_values.values())
        if len(explored_hls_results) != len(filtered_explored_values): return []
        all_options = [str(v) for v in self.ds_config[pragma_name] 
            if str(v) not in explored_values.keys() and str(v) != str(best_design[pragma_name])]
        num_updates = NUM_OPTIMIZATIONS if pragma_type != "pipeline" else 1
        if len(all_options) <= num_updates: return[(best_design, pragma_name, str(v)) for v in all_options]
        if AUTO_OPTIMIZER: return [(best_design, pragma_name, str(v)) for v in all_options[:num_updates]]
        try:
            update_prompt = compile_pragma_update_prompt(best_design, hls_results, pragma_name, self.c_code, all_options, pragma_type, warnings.get(pragma_name, []), explored_values, self.optimizer_reflections[pragma_name])
            return [(best_design, pragma_name, str(update.get(pragma_name))) for update in retrieve_list_from_response(get_openai_response(update_prompt))]
        except Exception as e:
            return [(best_design, pragma_name, str(v)) for v in all_options[:num_updates]]
    
    def select_best_update(self, pragma_updates: List[Tuple[dict, str, str]]) -> Tuple[dict, str, str]:
        if len(pragma_updates) <= NUM_CHOSENS: return pragma_updates
        if AUTO_ARBITRATOR: return random.sample(pragma_updates, NUM_CHOSENS)
        try:
            prompt = compile_arbitrator_prompt(self.c_code, pragma_updates)
            return [pragma_updates[_idx] for _idx in retrieve_indices_from_response(get_openai_response(prompt))]
        except Exception as e:
            return random.sample(pragma_updates, NUM_CHOSENS)

    def explore(self):
        pragma_updates: List[Tuple[dict, str, str]] = []
        for pragma_name in self.pragma_names:
            for idx in self.select_best_designs(pragma_name):
                pragma_updates.extend(self.propose_update(idx, pragma_name))
        return self.select_best_update(pragma_updates)

    def self_reflection(self, prev_design: dict, curr_design: dict, 
            prev_hls_results: Dict[str, str], prev_pragma_warnings: Dict[str, List[str]],
            curr_hls_results: Dict[str, str], curr_pragma_warnings: Dict[str, List[str]]):
        if not prev_hls_results or not curr_hls_results: return
        if AUTO_REFLECTION: return
        try:
            reflection_prompt = compile_reflection_prompt(self.c_code, prev_design, curr_design, 
                prev_hls_results, prev_pragma_warnings, curr_hls_results, curr_pragma_warnings, self.pragma_names)
            for pragma_name, reflections in retrieve_dict_from_response(get_openai_response(reflection_prompt)).items():
                print(f"Reflections for {pragma_name}:\n\t" + "\n\t".join(reflections))
                if pragma_name in self.optimizer_reflections:
                    self.optimizer_reflections[pragma_name].extend(reflections)
                    if len(self.optimizer_reflections[pragma_name]) > SELF_REFLECTION_LENGTH:
                        self.optimizer_reflections[pragma_name] = self.optimizer_reflections[pragma_name][-SELF_REFLECTION_LENGTH:]
        except Exception as e:
            return