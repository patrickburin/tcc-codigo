import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Carrega os dados da medição
X = np.load("measure_X.npy")
y = np.load("measure_y.npy")

# Converte o vetor one-hot [1,0,0,0,0,0] ou [0,0,0,0,0,1] em 1 (certo) ou 0 (errado)
# Assumimos que a primeira posição representa "certo"
y = np.argmax(y, axis=1)
y = np.where(y == 0, 1, 0)  # Se era a classe 0 (certo), vira 1; senão, vira 0

# Divide entre treino e teste (90/10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Cria o modelo LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Saída binária: 0 = errado, 1 = certo

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treina o modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)

# Mostra a acurácia final
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Acurácia final na base de teste: {acc * 100:.2f}%")
