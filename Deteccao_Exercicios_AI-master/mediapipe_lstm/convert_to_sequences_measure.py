import os
import json
import numpy as np

INPUT_DIR = "output_json"
SEQUENCE_LENGTH = 30

X = []
y = []

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue
    if "outros" in filename.lower():
        continue
    
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r") as f:
        frames = json.load(f)

    if len(frames) < SEQUENCE_LENGTH:
        continue

    for i in range(len(frames) - SEQUENCE_LENGTH + 1):
        sequence = frames[i:i+SEQUENCE_LENGTH]
        X.append(sequence)

        if "_noisy" in filename:
            y.append([0, 1])
        else:
            y.append([1, 0])

X = np.array(X)
y = np.array(y)

np.save("X_measure.npy", X)
np.save("y_measure.npy", y)

print(f"✔ Salvo {len(X)} sequências em 'X_measure.npy' e 'y_measure.npy'")
print(f"Formato X: {X.shape}, y: {y.shape}")
