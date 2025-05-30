import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import clone_model

# Carrega os dados
X = np.load("measure_X.npy")
y = np.load("measure_y.npy")

# Converte one-hot para binário (0 = erro, 1 = certo)
y = np.argmax(y, axis=1)
y = np.where(y == 0, 1, 0)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
best_model = None
best_acc = 0.0

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(f"\n🔁 Treinando Fold {fold + 1}")
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 Acurácia Fold {fold + 1}: {acc * 100:.2f}%")
    accuracies.append(acc)

    if acc > best_acc:
        best_acc = acc
        best_model = clone_model(model)
        best_model.set_weights(model.get_weights())

# Salva o melhor modelo
if best_model:
    best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    best_model.save("measure_final.keras")

print(f"\n✅ Acurácia média (Cross-validation): {np.mean(accuracies) * 100:.2f}%")
print(f"💾 Modelo com melhor acurácia ({best_acc * 100:.2f}%) salvo como 'measure_final.keras'")
