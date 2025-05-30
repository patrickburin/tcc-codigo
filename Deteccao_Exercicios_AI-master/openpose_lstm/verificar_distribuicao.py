import numpy as np
import os

# Agora sim com caminho relativo correto
detection_y = np.load(os.path.join("LSTM", "detection_y.npy"))

labels = np.argmax(detection_y, axis=1)

print("\n📊 Distribuição das classes no detection_y.npy:")
print(f"  ➤ Agachamento      [0]: {np.sum(labels == 0)} amostras")
print(f"  ➤ Extensão Quadril [1]: {np.sum(labels == 1)} amostras")
print(f"  ➤ Flexão Joelho    [2]: {np.sum(labels == 2)} amostras")
