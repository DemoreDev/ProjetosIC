from pathlib import Path
import sys

BASE_DIR  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import argparse
import csv
import os
import uuid
import arff
from src.process_arff import save_arff, read_arff, get_num_labels
import src.path_config as cfg
from src.java_executor import MekaExecutor
from src.output_parser import parse_output
from src.csv_translator import PipelineTranslator
import warnings

# Silencia os RuntimeWarnings do Scikit-Learn 
warnings.filterwarnings("ignore", category=RuntimeWarning)

import mlfs.br_skb
import mlfs.br_relieff
import mlfs.d2f_adapted
import mlfs.igmf_adapted
import mlfs.lrfs_adapted
import mlfs.lsmfs_adapted
import mlfs.mdmr_adapted
import mlfs.mlsmfs_adapted
import mlfs.pmu_adapted
import mlfs.ppt_mi_adapted
import mlfs.ppt_relieff
import mlfs.ppt_rfe
import mlfs.ppt_sfm
import mlfs.ppt_skb
import mlfs.scls_adapted

import sklearn.feature_selection

def initialize_output_csv():
    # Cria o arquivo CSV de saída com os cabeçalhos se ele não existir

    if not OUTPUT_CSV.exists():
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True) # Garante que a pasta existe
        
        header = [
            "mlc", "slc", "fp",                   # O Pipeline
            "execution_time",                     # Tempo de execução
            "real_f1", "real_size",               # Métricas Reais
            "predicted_F1",                       # F1 predito
            "predicted_model_size",               # Tamanho predito
            "status", "error"                     # Controle de Sucesso/Falha
        ]

        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_line(line: dict):
    # Adiciona uma única linha ao final do arquivo CSV

    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=line.keys())
        writer.writerow(line)


def main(args):
    print("Iniciando Orquestrador de Experimentos...")

        # 1. Escolha o primeiro arquivo de treino para servir de referência
    ref_arff = ARFF_PATH / f"{args.dataset_name.lower()}-train-0.arff"

    # 2. Descobre o número de labels 
    n_labels = get_num_labels(str(ref_arff))

    # 3. Descobre o número de features (Total de atributos - n_labels)
    total_attributes = 0
    with open(ref_arff, "r") as f:
        for line in f:
            if line.strip().lower().startswith("@attribute"):
                total_attributes += 1
    n_features = total_attributes - n_labels

    # Prepara o arquivo de saída
    initialize_output_csv()

    # Inicializa o Executor do Java
    executor = MekaExecutor(memory="8G", timeout_sec=3600)

    # Inicializa o tradutor CSV -> string
    translator = PipelineTranslator(n_features=n_features, n_labels=n_labels)

    # Carrega o CSV de entrada 
    #df = pd.read_csv(OTHER_PATH)
    df = pd.read_csv(CSV_PATH)

    # Garante que o valor passado não seja maior que o df
    top_n = min(args.top_n, len(df)) 
    
    # Loop principal (itera pelas linhas do CSV)
    for i in range(top_n):
        print(f"\nValidando pipeline [{i + 1}/{top_n}]...")

        # Pega a linha atual (removendo colunas de metadados se necessário)
        actual_line = df.iloc[i, 2:-2] 
        predicted_f1 = df.iloc[i, -3]
        predicted_model_size = df.iloc[i, -1]

        # 1. TRADUÇÃO: Agora recebe variáveis discretas e parâmetros já "desindexados" (ex: 0.6 em vez de 10)
        fp_cmd, meka_algo, meka_p, weka_algo, weka_p = translator.translate_row(actual_line)

        # Criamos um dicionário apenas para facilitar o log e o salvamento no CSV
        pipeline_repr = {
            "mlc": f"{meka_algo} {meka_p}",
            "slc": f"{weka_algo} {weka_p}",
            "fp": fp_cmd
        }
        
        folds_results = []
        total_time = 0 

        # Loop secundário (itera sobre os 3 folds)
        for fold in range(3):
            print(f"\nExecutando Fold {fold}...")
            
            orig_train_path = ARFF_PATH / f"{args.dataset_name.lower()}-train-{fold}.arff"
            orig_test_path  = ARFF_PATH / f"{args.dataset_name.lower()}-test-{fold}.arff"

            java_train_path = orig_train_path
            java_test_path = orig_test_path
            
            try:
                # 2. FEATURE PREPROCESSING (FP)
                if fp_cmd and str(fp_cmd).strip() != "" and str(fp_cmd) != "None":
                    print(f"    [+] Aplicando FP: {fp_cmd}")
                    
                    num_labels = get_num_labels(orig_train_path)
                    feat_types, dfX_train, dfy_train = read_arff(str(orig_train_path), num_labels)
                    _, dfX_test, dfy_test = read_arff(str(orig_test_path), num_labels)
                    
                    fp_algorithm = eval(fp_cmd)
                    fp_algorithm.fit(dfX_train, dfy_train)
                    
                    dfX_train_new = fp_algorithm.transform(dfX_train)
                    dfX_test_new = fp_algorithm.transform(dfX_test)

                    java_train_path = TEMP_DIR / f"temp_train_{uuid.uuid4().hex}.arff"
                    java_test_path = TEMP_DIR / f"temp_test_{uuid.uuid4().hex}.arff"

                    save_arff(dfX_train_new, dfy_train, feat_types, num_labels, args.dataset_name, str(java_train_path))
                    save_arff(dfX_test_new, dfy_test, feat_types, num_labels, args.dataset_name, str(java_test_path))

                # 3. MONTAGEM DO COMANDO: Usando a nova lógica que trata parâmetros reais e booleanos
                cmd, temp_model_path = executor.build_command(
                    meka_algo=meka_algo,
                    meka_params=meka_p,
                    weka_algo=weka_algo,
                    weka_params=weka_p,
                    train_path=str(java_train_path),
                    test_path=str(java_test_path)
                )
                
                # 4. EXECUÇÃO
                print("    [>] Executando Java...")
                res = executor.execute(cmd, temp_model_path, pipeline_repr)
                total_time += res.get("time_sec", 0)

                if res["success"]:
                    text_metrics = parse_output(res["output"])
                    f1 = text_metrics.get("f1_real")
                    size = res.get("model_size")
                    print(f"    [+] Sucesso! F1: {f1} | Tempo: {res.get('time_sec')}s")

                    # Log de Debug
                    debug_filepath = DEBUG_PATH / f"pipe{i+1}_fold{fold}.txt"
                    with open(debug_filepath, "w", encoding="utf-8") as f:
                        f.write(f"COMANDO: {' '.join(cmd)}\n\n")
                        f.write(f"SAIDA:\n{res['output']}")
                    
                    folds_results.append({"f1": f1, "size": size})
                else:
                    print(f"    [!] ERRO NO JAVA: {res.get('error', 'Erro desconhecido')}")

            except Exception as e:
                print(f"    [!] ERRO CRÍTICO NO PYTHON (Fold {fold}): {str(e)}")
                break

            finally:
                # Limpeza de arquivos temporários
                if java_train_path != orig_train_path and os.path.exists(java_train_path):
                    os.remove(java_train_path)
                if java_test_path != orig_test_path and os.path.exists(java_test_path):
                    os.remove(java_test_path)
        
        # 5. CONSOLIDAÇÃO DOS RESULTADOS (Média entre os folds)
        csv_line = {
            "mlc": meka_algo,
            "slc": weka_algo,
            "fp": fp_cmd,
            "execution_time": round(total_time, 2),
            "real_f1": 0.0,
            "real_size": 1e9,
            "predicted_F1": predicted_f1,
            "predicted_model_size": predicted_model_size,
            "status": "FALHA"
        }

        folds_validos = [r for r in folds_results if r["f1"] is not None]

        if len(folds_validos) > 0:
            import statistics
            f1s = [r["f1"] for r in folds_validos]
            sizes = [r["size"] for r in folds_validos]
            csv_line["real_f1"] = statistics.median(f1s)
            csv_line["real_size"] = statistics.median(sizes)
            csv_line["status"] = "SUCESSO"

        save_line(csv_line)
        print(f"-> Finalizado! Status: {csv_line['status']} | F1 Médio: {csv_line['real_f1']:.4f}")

    print("\nExperimentos finalizados com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para validar o desempenho de pipelines candidatos")

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['birds', 'medical', 'enron', 'scene', 'yeast'], 
        help="Nome do dataset que os pipelines serão validados (implementados e testados)"
    )

    parser.add_argument(
        '--top_n',
        type=int,
        required=True,
        help="Quantidade de pipelines (linhas) a serem validados"
    )

    args = parser.parse_args()

    # Configurações de Caminho
    CSV_PATH = BASE_DIR / "results" / "predicted_pipeline_ranking" / f"best_{args.dataset_name.lower()}_xgboost.csv"
    OUTPUT_CSV = BASE_DIR / "results" / "validation" / f"validated_{args.dataset_name.lower()}_pipelines.csv"
    DEBUG_PATH = BASE_DIR / "debug"
    DEBUG_PATH.mkdir(parents=True, exist_ok=True)
    ARFF_PATH = BASE_DIR / "data" / "raw" / f"{args.dataset_name.lower()}"
    TEMP_DIR = BASE_DIR / "temp"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    #OTHER_PATH = BASE_DIR / "data" / "meta" / "meta_processed" / "test_medical.csv"

    main(args)