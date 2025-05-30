import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import save_model

X = np.load("X_measure.npy")
y = np.load("y_measure.npy")
y_class = np.argmax(y, axis=1)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_acc = 0.0
best_model = None
fold = 1

for train_index, test_index in kf.split(X, y_class):
    print(f"\n🔁 Treinando fold {fold}/5...")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(30, 132)))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✔ Fold {fold} - Acurácia: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model

    fold += 1

save_model(best_model, "measure_classifier_best_cv.keras")
print(f"\n🏁 Melhor modelo salvo como 'measure_classifier_best_cv.keras' com acurácia de {best_acc:.4f}")
