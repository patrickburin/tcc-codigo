import os
import cv2

# Pastas dos exercícios (nível raiz)
pastas_exercicios = ["agachamento", "extensaoquadril", "flexaojoelho"]

# Pasta de saída dos frames crus
output_base = os.path.join("LSTM", "frames_videos")
os.makedirs(output_base, exist_ok=True)

for pasta in pastas_exercicios:
    caminho_pasta = os.path.join(".", pasta)
    if not os.path.isdir(caminho_pasta):
        continue

    for arquivo in os.listdir(caminho_pasta):
        if not arquivo.lower().endswith(".mp4"):
            continue

        caminho_video = os.path.join(caminho_pasta, arquivo)
        nome_base = os.path.splitext(arquivo)[0]
        nome_saida = f"{pasta}_{nome_base}"  # Ex: agachamento_agachamento01
        pasta_saida = os.path.join(output_base, nome_saida)
        os.makedirs(pasta_saida, exist_ok=True)

        print(f"\n🎞️ Extraindo frames de: {caminho_video}")
        cap = cv2.VideoCapture(caminho_video)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(pasta_saida, f"frame_{frame_idx:012d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1

        cap.release()
        print(f"✅ {frame_idx} frames salvos em: {pasta_saida}")
