import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import Counter
from tkinter import messagebox

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Selecione um vídeo",
    filetypes=[("Arquivos de vídeo", "*.mp4")]
)

if not file_path:
    print("❌ Nenhum vídeo foi selecionado.")
    exit()

print(f"✔ Vídeo selecionado: {file_path}")

cap = cv2.VideoCapture(file_path)
landmark_sequences = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        frame_landmarks = []
        for lm in results.pose_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        landmark_sequences.append(frame_landmarks)

cap.release()

SEQUENCE_LENGTH = 30
sequences = []
for i in range(len(landmark_sequences) - SEQUENCE_LENGTH + 1):
    seq = landmark_sequences[i:i + SEQUENCE_LENGTH]
    sequences.append(np.array(seq))

if not sequences:
    print("❌ Não foi possível criar sequências suficientes para inferência.")
    exit()

sequences = np.array(sequences)

det_model = load_model("exercise_classifier_best_cv.keras")
det_predictions = det_model.predict(sequences)
det_labels = np.argmax(det_predictions, axis=1)

label_map = {
    0: "Agachamento",
    1: "Extensão de Quadril",
    2: "Flexão de Joelho",
    3: "Outro"
}

det_count = Counter(det_labels)
most_common_label, most_common_count = det_count.most_common(1)[0]
det_accuracy = most_common_count / len(det_labels)

val_model = load_model("measure_classifier_best_cv.keras")
val_predictions = val_model.predict(sequences)
val_accuracy = float(np.mean(val_predictions[:, 0]))

print(f"\n🧠 Análise Combinada do Vídeo:")
print(f"📌 Exercício detectado: {label_map[most_common_label]}")
print(f"📊 Confiabilidade da detecção: {det_accuracy:.4f} ({most_common_count}/{len(det_labels)} sequências)")

message = f"🧠 Análise Combinada do Vídeo:\n\n"
message += f"📌 Exercício detectado: {label_map[most_common_label]}\n"
message += f"📊 Confiabilidade da detecção: {det_accuracy:.4f} ({most_common_count}/{len(det_labels)} sequências)\n"

show_bar = False
if det_accuracy >= 0.75 and most_common_label != 3:
    show_bar = True
    percent = val_accuracy * 100
    if percent >= 90:
        message += f"✅ Execução classificada como CORRETA com {percent:.2f}% de confiança.\n"
    elif percent >= 75:
        message += f"⚠ Execução correta, mas precisa de ajustes ({percent:.2f}%).\n"
    else:
        message += f"❌ Execução classificada como INCORRETA ({percent:.2f}%).\n"
    print(message.split('\n')[-2])  # imprime a última linha da mensagem
else:
    message += "⚠ Não foi possível avaliar a execução: detecção inconclusiva ou exercício fora do padrão."
    print("⚠ Não foi possível avaliar a execução: detecção inconclusiva ou exercício fora do padrão.")


from tkinter import Toplevel, Label, Button, Text, Scrollbar, RIGHT, Y, END, Canvas

def exibir_resultado(mensagem, barra=None):
    root = tk.Tk()
    root.title("Resultado da Análise")
    root.geometry("500x350")
    root.resizable(False, False)

    text_area = Text(root, wrap="word", font=("Arial", 12), padx=10, pady=10, height=10)
    text_area.insert(END, mensagem)
    text_area.config(state="disabled")
    text_area.pack(expand=False, fill="x")

    if barra:
        canvas = Canvas(root, width=400, height=30)
        canvas.pack(pady=10)

        cor = "green" if barra >= 90 else "orange" if barra >= 75 else "red"
        canvas.create_rectangle(0, 0, barra * 4, 30, fill=cor)
        canvas.create_text(200, 15, text=f"{barra:.2f}%", fill="white", font=("Arial", 12, "bold"))

    btn_fechar = Button(root, text="Fechar", font=("Arial", 11), command=root.destroy)
    btn_fechar.pack(pady=10)

    root.mainloop()

exibir_resultado(message, val_accuracy * 100 if show_bar else None)
