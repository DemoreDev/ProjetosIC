import subprocess
import time
import uuid 
import os
import json
import numpy as np
from pathlib import Path
import src.path_config as cfg

# Define a raiz do projeto 
BASE_DIR = Path(__file__).resolve().parent.parent

# Define o caminho da pasta temp
TEMP_DIR = BASE_DIR / "temp"

# Garante que a pasta existe 
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class MekaExecutor:
    # Responsável por formatar e executar comandos do Meka/Weka via terminal (Java).
    
    def __init__(self, lib_path: Path = None, memory: str = "8G", timeout_sec: int = 3600):
        # Caso libpath seja passado
        if lib_path is not None:
            self.classpath = f"{lib_path}/*"
        else:
            self.classpath = cfg.MEKA_CLASSPATH
        
        self.memory = f"-Xmx{memory}"
        self.timeout = timeout_sec

    def build_command(self, meka_algo, meka_params, weka_algo, weka_params, train_path, test_path) -> tuple:
        """
        Agora recebe algoritmos e dicionários de parâmetros diretamente.
        """
        if not meka_algo:
            raise ValueError("O algoritmo MLC (Meka) não pode ser vazio.")

        # 1. Base do Comando Java
        cmd = ["java", self.memory, "-cp", self.classpath]

        # 2. Classe Principal e Parâmetros do Meka
        # Lógica especial para o MULAN (que usa -S)
        if ".MULAN." in meka_algo:
            parts = meka_algo.split(".MULAN.")
            cmd.append(f"{parts[0]}.MULAN")
            
            mulan_sub_algo = parts[1]
            # No MULAN, parâmetros viram sufixos: Algoritmo.P1.P2
            params_suffix = [str(v) for k, v in meka_params.items() if "normalize" not in k]
            if params_suffix:
                mulan_sub_algo = f"{mulan_sub_algo}.{'.'.join(params_suffix)}"
            
            cmd.extend(["-S", mulan_sub_algo])
            
            if meka_params.get("-normalize") or meka_params.get("normalize"):
                cmd.append("-normalize")
        else:
            cmd.append(meka_algo)
            # Adiciona parâmetros do Meka (ex: -P 0.6)
            for flag, val in meka_params.items():
                self._append_param(cmd, flag, val)

        # 3. Metadados e Caminhos (Sempre após o algoritmo principal)
        temp_model_path = TEMP_DIR / f"temp_model_{uuid.uuid4().hex}.model"
        cmd.extend([
            "-t", str(train_path),
            "-T", str(test_path),
            "-d", str(temp_model_path),
            "-verbosity", "3"
        ])

        # 4. Processa o WEKA (SLC)
        if weka_algo:
            # Caso especial: se for um Kernel, o Meka exige envolver no SMO
            if "supportVector" in weka_algo:
                cmd.extend(["-W", "weka.classifiers.functions.SMO", "--", "-K", weka_algo])
            else:
                cmd.extend(["-W", weka_algo, "--"])
            
            # Adiciona os parâmetros do Weka/Kernel
            for flag, val in weka_params.items():
                self._append_param(cmd, flag, val)

        print(f"\n[DEBUG EXECUTOR] Comando montado:\n{' '.join(cmd)}\n")
        return cmd, str(temp_model_path)

    def _append_param(self, cmd_list, flag, val):
        clean_flag = flag if flag.startswith("-") else f"-{flag}"
        
        if isinstance(val, (bool, np.bool_)):
            if bool(val) == True:
                cmd_list.append(clean_flag)
            return

        if val is not None and str(val).strip() not in ["", "None"]:
            # LIMPEZA CRÍTICA: Remove aspas simples que podem ter vindo do dict original
            # Isso evita o NullPointerException no payoff do MCC/PMCC
            clean_val = str(val).replace("'", "").replace('"', "")
            
            cmd_list.append(clean_flag)
            cmd_list.append(clean_val)
    

    def execute(self, command_list: list, temp_model_path: str, pipeline_info: dict = None) -> dict:
        # Executa o comando montado e captura a saída.
        
        start_time = time.time()
        pipeline_info = pipeline_info or {} 
        model_size_bytes = None
        
        print("[DEBUG EXECUTOR] Iniciando subprocess do java")

        try:
            result = subprocess.run(
                command_list,
                capture_output=True, 
                text=True,           
                timeout=self.timeout
            )
            
            print("[DEBUG EXECUTOR] subprocess do java finalizado!")
            elapsed_time = time.time() - start_time
            success = (result.returncode == 0)

            if success and os.path.exists(temp_model_path):
                # Pega o tamanho do arquivo em bytes 
                model_size_bytes = os.path.getsize(temp_model_path)
                # Deleta o modelo imediatamente
                os.remove(temp_model_path)
            
            # Pega o erro real. Se o stderr estiver vazio, tenta achar pistas no stdout
            error_desc = result.stderr.strip()
            if not success and not error_desc:
                error_desc = f"Erro no stdout ou processo morto (código {result.returncode}):\n{result.stdout.strip()}"

            response = {
                "success": success,
                "time_sec": round(elapsed_time, 2),
                "output": result.stdout,
                "error": error_desc if not success else "",
                "returncode": result.returncode,
                "model_size": model_size_bytes
            }
            
            if not success:
                self._log_failure(command_list, pipeline_info, response)
                
            return response
        
        # Se der timeout, salva os dados
        except subprocess.TimeoutExpired:
            print(f"      [!] TIMEOUT: O modelo demorou mais de {self.timeout}s e foi abortado.")
            elapsed_time = time.time() - start_time
            response = {
                "success": False,
                "time_sec": round(elapsed_time, 2),
                "output": "",
                "error": f"TIMEOUT: O processo excedeu {self.timeout} segundos.",
                "returncode": None
            }
            self._log_failure(command_list, pipeline_info, response)
            # Apaga um modelo temporário (caso tenha sido criado)
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            return response
            
        # Se deu erro por outro motivo
        except Exception as e:
            print(f"      [!] ERRO CRÍTICO NO PYTHON: {str(e)}")
            response = {
                "success": False,
                "time_sec": round(time.time() - start_time, 2),
                "output": "",
                "error": f"ERRO INTERNO: {str(e)}",
                "returncode": None
            }
            self._log_failure(command_list, pipeline_info, response)
            # Apaga um modelo temporário (caso tenha sido criado)
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            return response
        
        
    def _log_failure(self, command_list: list, pipeline_info: dict, execution_result: dict):
        """
        Salva o contexto completo do erro em um arquivo JSON para análise futura.
        """
        log_entry = {
            "command": " ".join(command_list), # Facilita copiar e colar no terminal para testar
            "pipeline_context": pipeline_info, # Aqui ficam as flags, hyperparams, etc.
            "execution_details": execution_result
        }
        
        # Salva em um arquivo com append (cada linha sendo um JSON válido ajuda na leitura posterior)
        with open("error_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")