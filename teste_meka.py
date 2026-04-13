import subprocess
import re
from pathlib import Path
import sys

BASE_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def extrair_f1(output):
    """Procura a linha do F1 Macro por label e extrai o valor numérico."""
    for linha in output.split('\n'):
        if "F1 (macro averaged by label)" in linha:
            # Captura o número no final da linha (funciona com ponto ou vírgula)
            match = re.search(r"(\d+[.,]\d+)", linha)
            if match:
                return float(match.group(1).replace(',', '.'))
    return None

f1_scores = []
data = "medical"

comando = [
        "java",
        "-Xmx8G",
        "-cp", f"{str(BASE_DIR)}/lib/*",
        "meka.classifiers.multilabel.BRq",
        "-t", f"{str(BASE_DIR)}/data/raw/medical/medical-train-2.arff", 
        "-T", f"{str(BASE_DIR)}/data/raw/medical/medical-test-2.arff",  
        "-P", "0.6",
        "-verbosity", "3",
        "-W", "weka.classifiers.trees.LMT",
        "--", 
        "-M", "41",
        "-W", "0.01",
        "-P",
        "-A"
    ]
"""
processo = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            check=True 
        )
print(processo.stdout)
"""

for i in range(3):
    print(f"\n--- INICIANDO FOLD {i} ---")
    
    comando2 = [
        "java",
        "-Xmx8G",
        "-cp", f"{str(BASE_DIR)}/lib/*",
        "meka.classifiers.multilabel.BRq",
        "-t", f"{str(BASE_DIR)}/data/raw/{data}/{data}-train-{i}.arff", 
        "-T", f"{str(BASE_DIR)}/data/raw/{data}/{data}-test-{i}.arff",  
        "-C", "45",
        "-d", f"{str(BASE_DIR)}/temp/temp_model_fold_{i}.model",
        "-P", "0.6",
        "-verbosity", "3",
        "-W", "weka.classifiers.trees.LMT",
        "--", 
        "-M", "41",
        "-W", "0.01",
        "-P",
        "-A"
    ]

    try:
        processo = subprocess.run(
            comando2,
            capture_output=True,
            text=True,
            check=True 
        )
        print(processo.stdout)

        """
        f1_fold = extrair_f1(processo.stdout)
        
        if f1_fold is not None:
            f1_scores.append(f1_fold)
            print(f"✅ Fold {i} concluído. F1: {f1_fold}")
        else:
            print(f"⚠️ Fold {i} concluído, mas o F1 não foi encontrado na saída.")

        # Salva a saída do último fold para conferência se desejar
        with open(f"saida_fold_{i}.txt", "w") as f:
            f.write(processo.stdout)
        """

    except subprocess.CalledProcessError as e:
        print(f"❌ ERRO NO FOLD {i}!")
        print(e.stderr)

"""
# --- CÁLCULO DA MÉDIA FINAL ---
print("\n" + "="*30)
if len(f1_scores) > 0:
    media_f1 = sum(f1_scores) / len(f1_scores)
    print(f"RESULTADO FINAL (Média de {len(f1_scores)} folds)")
    print(f"F1 Macro Médio: {media_f1:.4f}")
else:
    print("Não foi possível calcular a média (nenhum F1 extraído).")
print("="*30)
"""