import sys
import json
import math
import os
import glob

def convertDegreeToRadian(degree):
    return ((degree * math.pi) / 180)

def convertRadianToDegree(radian):
    return ((radian * 180) / math.pi)

def anglesForArms(data):
    if (data[1][0] == 0 and data[1][1] == 0) or (data[8][0] == 0 and data[8][1] == 0) or (data[3][0] == 0 and data[3][1] == 0):
        return 0
    ax = data[1][0] - data[3][0]
    ay = data[1][1] - data[3][1]
    a = math.sqrt(ax**2 + ay**2)
    bx = data[1][0] - data[8][0]
    by = data[1][1] - data[8][1]
    b = math.sqrt(bx**2 + by**2)
    cx = data[3][0] - data[8][0]
    cy = data[3][1] - data[8][1]
    c = math.sqrt(cx**2 + cy**2)
    cos0 = (c**2 - a**2 - b**2) / (-2 * a * b)
    inverse_cos0 = math.acos(cos0)
    return convertRadianToDegree(inverse_cos0)

def anglesForBackbone(data):
    if (data[1][0] == 0 and data[1][1] == 0) or (data[8][0] == 0 and data[8][1] == 0) or (data[10][0] == 0 and data[10][1] == 0):
        return 0
    ax = data[1][0] - data[8][0]
    ay = data[1][1] - data[8][1]
    a = math.sqrt(ax**2 + ay**2)
    bx = data[8][0] - data[10][0]
    by = data[8][1] - data[10][1]
    b = math.sqrt(bx**2 + by**2)
    cx = data[1][0] - data[10][0]
    cy = data[1][1] - data[10][1]
    c = math.sqrt(cx**2 + cy**2)
    cos0 = (c**2 - a**2 - b**2) / (-2 * a * b)
    inverse_cos0 = math.acos(cos0)
    return convertRadianToDegree(inverse_cos0)

def anglesForKnees(data):
    if (data[9][0] == 0 and data[9][1] == 0) or (data[10][0] == 0 and data[10][1] == 0) or (data[11][0] == 0 and data[11][1] == 0):
        return 0
    ax = data[9][0] - data[10][0]
    ay = data[9][1] - data[10][1]
    a = math.sqrt(ax**2 + ay**2)
    bx = data[10][0] - data[11][0]
    by = data[10][1] - data[11][1]
    b = math.sqrt(bx**2 + by**2)
    cx = data[9][0] - data[11][0]
    cy = data[9][1] - data[11][1]
    c = math.sqrt(cx**2 + cy**2)
    cos0 = (c**2 - a**2 - b**2) / (-2 * a * b)
    inverse_cos0 = math.acos(cos0)
    return convertRadianToDegree(inverse_cos0)

def anglesBetweenLegs(data):
    if (data[10][0] == 0 and data[10][1] == 0) or (data[12][0] == 0 and data[12][1] == 0) or (data[13][0] == 0 and data[13][1] == 0):
        return 0
    ax = data[12][0] - data[10][0]
    ay = data[12][1] - data[10][1]
    a = math.sqrt(ax**2 + ay**2)
    bx = data[12][0] - data[13][0]
    by = data[12][1] - data[13][1]
    b = math.sqrt(bx**2 + by**2)
    cx = data[10][0] - data[13][0]
    cy = data[10][1] - data[13][1]
    c = math.sqrt(cx**2 + cy**2)
    cos0 = (c**2 - a**2 - b**2) / (-2 * a * b)
    inverse_cos0 = math.acos(cos0)
    return convertRadianToDegree(inverse_cos0)

def processar_jsons_em_pasta(pasta_jsons, nome_saida):
    import cv2

    all_angles = []
    total_valid_frames = 0
    arquivos = sorted(glob.glob(os.path.join(pasta_jsons, "*.json")))

    # Nova pasta dentro de LSTM para os frames
    pasta_frames_output = os.path.join("LSTM", "output_frames_openpose")
    os.makedirs(pasta_frames_output, exist_ok=True)

    for idx, arquivo in enumerate(arquivos):
        with open(arquivo) as f:
            data = json.load(f)
            if not data["people"]:
                continue

            total_valid_frames += 1
            all_data = data["people"][0]
            keypoints = all_data["pose_keypoints_2d"]
            points_array = []
            for i in range(0, len(keypoints), 3):
                x, y = keypoints[i], keypoints[i + 1]
                points_array.append([x, y])
            angles_per_frame = [
                0.0, 0.0, 0.0,  # Dummy placeholders
                anglesForArms(points_array),
                anglesForBackbone(points_array),
                anglesForKnees(points_array),
                anglesBetweenLegs(points_array)
            ]
            all_angles.append(angles_per_frame)

            # Tenta encontrar o frame correspondente com base no nome
            frame_number = idx  # Assume que os JSONs estão ordenados
            frame_path = os.path.join("video_frames", f"frame_{frame_number:012d}.jpg")
            if os.path.isfile(frame_path):
                frame = cv2.imread(frame_path)
                frame_out_path = os.path.join(pasta_frames_output, f"frame_{frame_number:04d}.jpg")
                cv2.imwrite(frame_out_path, frame)

    print(f"\n📊 Total de frames válidos com pessoas detectadas: {total_valid_frames}")
    with open(nome_saida, 'w') as f:
        json.dump(all_angles, f)


# ✅ Chamando a função quando executado como script
if __name__ == "__main__":
    processar_jsons_em_pasta("LSTM/openpose_output", "all_features.json")
