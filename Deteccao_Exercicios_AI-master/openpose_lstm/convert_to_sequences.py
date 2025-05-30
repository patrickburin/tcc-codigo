import os
import json
import numpy as np

# Caminho base (sobe um nível da pasta LSTM)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Configuração da janela
SEQUENCE_LENGTH = 30

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_npy(obj, name):
    np.save(os.path.join(os.path.dirname(__file__), name), obj)

def slice_sequences(data, sequence_length):
    return [data[i:i+sequence_length] for i in range(0, len(data) - sequence_length + 1, sequence_length)]

# ---------- DETECÇÃO ----------
detection_X = []
detection_y = []

for filename in os.listdir(BASE_DIR):
    if not (filename.startswith("features_") and filename.endswith(".json")):
        continue

    if "extensaoquadril" in filename:
        label = [1, 0]
    elif "flexaojoelho" in filename:
        label = [0, 1]
    else:
        continue  # ignora arquivos que não são dessas duas classes

    filepath = os.path.join(BASE_DIR, filename)
    data = load_json(filepath)
    sequences = slice_sequences(data, SEQUENCE_LENGTH)
    detection_X.extend(sequences)
    detection_y.extend([label] * len(sequences))

print(f"[DETECÇÃO] Total sequências: {len(detection_X)}")
save_npy(np.array(detection_X), "detection_X.npy")
save_npy(np.array(detection_y), "detection_y.npy")

def converter_um_arquivo(filepath):
    data = load_json(filepath)

    # Novo formato (lista): avaliação de vídeo com um único conjunto de frames
    if isinstance(data, list):
        sequences = slice_sequences(data, 30)
        print(f"[DEBUG] Total de sequências detectadas no JSON: {len(sequences)}")

        # Ajusta: detecção usa só os 4 últimos valores (angulos), medição usa todos os 7
        detect_sequences = [[frame[-4:] for frame in seq] for seq in sequences]
        measure_sequences = sequences  # todos os 7 valores

        return np.array(detect_sequences), np.array(measure_sequences)

    # Formato antigo (dicionário): treino/validação
    detection_X = []
    detection_y = []
    measure_X = []
    measure_y = []

    for label_idx, (exercicio, frames) in enumerate(data.items()):
        sequences = slice_sequences(frames, 30)
        detection_X.extend([[frame[-4:] for frame in seq] for seq in sequences])

        if exercicio == "extensaoquadril":
            label = [1, 0]
        elif exercicio == "flexaojoelho":
            label = [0, 1]
        else:
            continue

        detection_y.extend([label] * len(sequences))

        measure_X.extend(sequences)
        measure_y.extend([0.0] * len(sequences))  # placeholder

    print(f"[DETECÇÃO] Total sequências: {len(detection_X)}")
    print(f"[MEDIÇÃO] Total sequências: {len(measure_X)}")

    return np.array(detection_X), np.array(measure_X)

# ---------- MEDIÇÃO ----------
measure_X = []
measure_y = []

for filename in os.listdir(BASE_DIR):
    if filename.startswith("output_noisy_features_") and filename.endswith(".json"):
        filepath = os.path.join(BASE_DIR, filename)
        raw_data = load_json(filepath)

        # extrai [classe + 4 ângulos] → 7 elementos por frame
        data = [[*frame[0], frame[4], frame[5], frame[6], frame[7]] for frame in raw_data]
        errors = [frame[8] for frame in raw_data]

        sequences = slice_sequences(data, SEQUENCE_LENGTH)
        error_labels = slice_sequences(errors, SEQUENCE_LENGTH)

        measure_X.extend(sequences)
        # pega o vetor de erro do último frame da sequência
        measure_y.extend([seq[-1] for seq in error_labels])

print(f"[MEDIÇÃO] Total sequências: {len(measure_X)}")
save_npy(np.array(measure_X), "measure_X.npy")
save_npy(np.array(measure_y), "measure_y.npy")
