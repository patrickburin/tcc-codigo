import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Carrega os dados
X = np.load("detection_X.npy")
y = np.load("detection_y.npy")

# K-fold (5 divisões, mantendo a proporção das classes)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
best_model = None
best_accuracy = 0

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, np.argmax(y, axis=1))):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\n🔁 Treinando Fold {fold + 1}")
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 Acurácia Fold {fold + 1}: {acc * 100:.2f}%")
    accuracies.append(acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        print("✅ Novo melhor modelo salvo.")

# Salva o melhor modelo após todos os folds
best_model.save("detection_final.keras")
print(f"\n✅ Acurácia média (Cross-validation): {np.mean(accuracies) * 100:.2f}%")
