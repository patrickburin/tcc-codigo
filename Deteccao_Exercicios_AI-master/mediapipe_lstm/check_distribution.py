import numpy as np
y = np.load("y_measure.npy")
labels = np.argmax(y, axis=1)
_, counts = np.unique(labels, return_counts=True)
print(f"Corretos: {counts[0]} | Incorretos: {counts[1]}")
