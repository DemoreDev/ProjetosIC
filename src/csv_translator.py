import pandas as pd
import numpy as np
import src.path_config as cfg
from src.search_space import get_mlc, get_slc, get_kernels_smo


class PipelineTranslator:
    def __init__(self, n_features: int, n_labels: int, param_types_json: dict = None):
        self.n_features = n_features
        self.n_labels = n_labels
        
        # Carrega os tipos básicos (bool, float, int) do seu JSON original
        self.param_types = param_types_json if param_types_json else cfg.load_hyperparameters_json()
        
        # Gera os espaços de busca REAIS baseados no dataset atual
        self.mlc_spaces = get_mlc(n_features, n_labels)
        self.slc_spaces = get_slc(n_features, n_labels)
        self.kernel_spaces = get_kernels_smo()


    def translate_row(self, csv_row: pd.Series) -> tuple:
        active_algos = self._get_active_algorithms(csv_row)
        fp_algo, meka_algo, weka_algo = self._categorize_algorithms(active_algos)
        fp_params, meka_params, weka_params = self._extract_params(active_algos, csv_row)

        # Mapeia os índices para valores REAIS usando o dicts.py
        meka_params_final = self._apply_search_space(meka_algo, meka_params, self.mlc_spaces)
        weka_params_final = self._apply_search_space(weka_algo, weka_params, self.slc_spaces)

        # PRINT DE DEBUG
        print(f"[DEBUG TRANSLATOR] Algo Extraído -> FP: {fp_algo}")
        print(f"[DEBUG TRANSLATOR] Params Extraídos -> FP: {fp_params}")

        print(f"\n[DEBUG TRANSLATOR] Algo Extraído -> MEKA: {meka_algo}")
        print(f"[DEBUG TRANSLATOR] Params Extraídos -> MEKA: {meka_params_final}")

        print(f"\n[DEBUG TRANSLATOR] Algo Extraído -> WEKA: {weka_algo}")
        print(f"[DEBUG TRANSLATOR] Params Extraídos -> WEKA: {weka_params_final}")

        # FP continua como string pois é executado via eval() no Python geralmente
        fp_command = self._build_fp_string(fp_algo, fp_params)

        # Retornamos os objetos estruturados para o Executor
        return fp_command, meka_algo, meka_params_final, weka_algo, weka_params_final
    

    def _apply_search_space(self, algo: str, params: dict, space_dict: dict) -> dict:
        if not algo or algo not in space_dict:
            return params

        config = space_dict[algo]
        mapped = {}

        # 1. Mapeamento Direto
        for flag, val in params.items():
            clean_flag = f"-{flag}" if not flag.startswith("-") else flag
            
            if clean_flag in config:
                space = config[clean_flag]
                try:
                    idx = int(float(val))
                    idx = idx % len(space) 
                    
                    real_val = space[idx]
                    # Converte np.bool_ para bool nativo para facilitar a vida do Executor
                    if isinstance(real_val, (bool, np.bool_)):
                        mapped[clean_flag] = bool(real_val)
                    else:
                        mapped[clean_flag] = real_val
                except:
                    mapped[clean_flag] = val
            else:
                mapped[clean_flag] = val

        # 2. Lógica de Condicionais (REMOVIDO O -1)
        if 'if' in config:
            extra_config = config['if'](mapped)
            if extra_config:
                for ex_flag, ex_space in extra_config.items():
                    csv_flag = ex_flag.lstrip("-") 
                    
                    if csv_flag in params:
                        try:
                            # Agora usa a mesma lógica de Base 0 do mapeamento direto
                            raw_idx = int(float(params[csv_flag]))
                            idx = raw_idx % len(ex_space)
                            
                            real_val = ex_space[idx]
                            if isinstance(real_val, (bool, np.bool_)):
                                mapped[ex_flag] = bool(real_val)
                            else:
                                mapped[ex_flag] = real_val
                        except: 
                            pass
        return mapped


    def _get_active_algorithms(self, csv_row: pd.Series) -> list:
        
        algorithm_cols = [col for col in csv_row.index if '-' not in col]

        return [algo for algo in algorithm_cols if csv_row[algo] == 1]
    

    def _categorize_algorithms(self, active_algos: list) -> tuple:

        fp, meka, weka = None, None, None
        
        for algo in active_algos:
            if "mlfs" in algo or "sklearn" in algo.lower():
                fp = algo
            elif "meka.classifiers" in algo:
                meka = algo
            elif "weka.classifiers" in algo:
                weka = algo
            else:
                print(f"Erro: coluna '{algo}' não pertence à fp, meka ou weka")

        # PRINT DE DEBUG
        return fp, meka, weka
    

    def _extract_params(self, algos: list, csv_row: pd.Series) -> tuple:
        # Inicializa dicionários vazios
        fp_params, meka_params, weka_params = {}, {}, {}

        for algo in algos:
            params = {}
            prefix = f"{algo}-"
            param_cols = [col for col in csv_row.index if str(col).startswith(prefix)]

            for col in param_cols:
                val = csv_row[col]
                
                # Ignora parâmetros inativos (-1)
                if val != -1:
                    flag_name = str(col).split('-')[-1]
                    
                    param_info = self.param_types.get(col, {})
                    
                    param_type = param_info.get("type", "float") # Float como padrão de segurança

                    if param_type == "bool":
                        if val == 1:
                            params[flag_name] = True
                    else:
                        # Limpa floats que são inteiros redondos
                        if isinstance(val, float) and val.is_integer():
                            params[flag_name] = int(val)
                        else:
                            params[flag_name] = val
            
            # Distribui os parâmetros extraídos para o dicionário correto
            if "mlfs" in algo or "sklearn" in algo.lower():
                fp_params = params
            elif "meka.classifiers" in algo:
                meka_params = params
            elif "weka.classifiers" in algo:
                weka_params = params
                        
        return fp_params, meka_params, weka_params


    
    
    def _build_fp_string(self, algo: str, params: dict) -> str:
        # Constrói a string para FP (python)
        if not algo:
            return ""

        # Mapeamentos baseados nos arrays de configuração 
        map_method = {
            0: "sklearn.feature_selection.f_classif",
            1: "sklearn.feature_selection.chi2",
            2: "sklearn.feature_selection.mutual_info_classif"
        }
        
        map_estimator = {
            0: "sklearn.ensemble.ExtraTreesClassifier()",
            1: "sklearn.ensemble.RandomForestClassifier()"
        }

        # Constrói a lista no formato 'chave=valor'
        formatted_params = []
        
        for flag, val in params.items():
            final_val = val
            
            # Traduz os índices para os valores reais das strings/objetos
            if flag == "method" and isinstance(val, int):
                final_val = map_method.get(val, val)
                
            elif flag == "estimator" and isinstance(val, int):
                final_val = map_estimator.get(val, val)
                
            elif flag == "neighbors" and isinstance(val, int):
                final_val = 10 if val == 0 else val

            # Adiciona na lista de parâmetros
            formatted_params.append(f"{flag}={final_val}")

        # Junta os parâmetros separando por vírgula e engloba nos parênteses
        params_str = ", ".join(formatted_params)
        
        return f"{algo}({params_str})"