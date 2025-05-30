import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 132)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✔ Acurácia no conjunto de teste: {acc:.4f}")

model.save("exercise_classifier_80_20.keras")
print("✔ Modelo salvo como 'exercise_classifier_80_20.keras'")
