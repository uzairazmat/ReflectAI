import os
import cv2
import json

def save_image(frame, timestamp, emotion):
    os.makedirs("emotion_logs/images", exist_ok=True)
    path = f"emotion_logs/images/{timestamp}_{emotion}.jpg"
    cv2.imwrite(path, frame)
    return path

def log_to_file(timestamp, emotion, confidence_scores, image_path):
    os.makedirs("emotion_logs", exist_ok=True)
    log_path = "emotion_logs/predictions.json"

    # Load safely
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print("⚠️ predictions.json is corrupted. Reinitializing.")
            data = {}
    else:
        data = {}

    data[timestamp] = {
        "emotion": emotion,
        "confidence": {k: round(float(v), 2) for k, v in confidence_scores.items()},
        "image_path": image_path
    }

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)
