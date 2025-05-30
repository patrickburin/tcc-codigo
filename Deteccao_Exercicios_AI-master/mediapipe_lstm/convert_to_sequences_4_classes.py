import os
import json
import numpy as np

INPUT_DIR = "output_json"
OUTPUT_X = "X.npy"
OUTPUT_Y = "y.npy"

SEQUENCE_LENGTH = 30

EXERCISE_LABELS = {
    "agachamento": [1, 0, 0, 0],
    "extensaoquadril": [0, 1, 0, 0],
    "flexaojoelho": [0, 0, 1, 0],
    "outros": [0, 0, 0, 1]
}

X = []
y = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        exercise_type = filename.split("_")[0]
        label = EXERCISE_LABELS.get(exercise_type)
        if not label:
            continue

        with open(os.path.join(INPUT_DIR, filename), "r") as f:
            frames = json.load(f)

        if len(frames) < SEQUENCE_LENGTH:
            continue

        for i in range(len(frames) - SEQUENCE_LENGTH + 1):
            sequence = frames[i:i + SEQUENCE_LENGTH]
            sequence_np = np.array(sequence).reshape(SEQUENCE_LENGTH, -1)
            X.append(sequence_np)
            y.append(label)

X = np.array(X)
y = np.array(y)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print(f"✔ Salvo {len(X)} sequências em '{OUTPUT_X}' e '{OUTPUT_Y}'")
print(f"Formato X: {X.shape}, y: {y.shape}")
