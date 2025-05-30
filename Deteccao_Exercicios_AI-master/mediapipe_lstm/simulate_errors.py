import os
import json
import random

INPUT_DIR = "output_json"
OUTPUT_DIR = "output_json"
OSCILACAO1 = 0.25
OSCILACAO = 0.5

def aplicar_erro_severo(ponto):
    return [
        coord + random.uniform(-OSCILACAO, OSCILACAO) for coord in ponto
    ]

gerados = 0

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue
    if "_noisy" in filename:
        continue

    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r") as f:
        sequencia = json.load(f)

    sequencia_ruidosa = []
    for frame in sequencia:
        ponto_com_erro = aplicar_erro_severo(frame)
        sequencia_ruidosa.append(ponto_com_erro)

    nome_saida = filename.replace(".json", "_noisy.json")
    path_saida = os.path.join(OUTPUT_DIR, nome_saida)

    with open(path_saida, "w") as f:
        json.dump(sequencia_ruidosa, f)

    print(f"✔ Gerado com erro: {path_saida}")
    gerados += 1

print(f"\n✔ Total de arquivos ruidosos gerados: {gerados}")
