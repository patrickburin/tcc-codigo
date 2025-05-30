import os
import cv2
import json
import numpy as np
import mediapipe as mp

BASE_PATH = "."

PASTAS_EXERCICIOS = ["agachamento", "extensaoquadril", "flexaojoelho", "outros"]

OUTPUT_JSON_PATH = "output_json"
OUTPUT_FRAMES_PATH = "output_frames"
os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)
os.makedirs(OUTPUT_FRAMES_PATH, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

for pasta in PASTAS_EXERCICIOS:
    pasta_path = os.path.join(BASE_PATH, pasta)
    if not os.path.isdir(pasta_path):
        continue

    for filename in os.listdir(pasta_path):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(pasta_path, filename)
        cap = cv2.VideoCapture(video_path)
        frames_landmarks = []

        frame_idx = 0
        nome_base = os.path.splitext(filename)[0]
        pasta_frames = os.path.join(OUTPUT_FRAMES_PATH, f"{pasta}_{nome_base}")
        os.makedirs(pasta_frames, exist_ok=True)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                frame_filename = os.path.join(pasta_frames, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

                frame_data = []
                for lm in results.pose_landmarks.landmark:
                    frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
                frames_landmarks.append(frame_data)

            frame_idx += 1

        cap.release()

        nome_json = f"{pasta}_{nome_base}.json"
        caminho_json = os.path.join(OUTPUT_JSON_PATH, nome_json)
        with open(caminho_json, "w") as f:
            json.dump(frames_landmarks, f)

        print(f"✔ Extraído: {caminho_json} | Frames salvos em: {pasta_frames}")
