import os
import cv2
import json

def save_image(frame, timestamp, emotion):
    os.makedirs("emotion_logs/images", exist_ok=True)
    path = f"emotion_logs/images/{timestamp}_{emotion}.jpg"
    cv2.imwrite(path, frame)
    return path


def log_to_file(timestamp, emotion, confidence_scores, image_path, fatigue_status="not detected"):
    os.makedirs("emotion_logs", exist_ok=True)
    log_path = "emotion_logs/detailed_predictions.json"  # Changed filename to avoid conflict

    # Load existing data
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = {}

    # Store complete data with fatigue
    data[timestamp] = {
        "emotion": emotion,
        "confidence": {k: round(float(v), 2) for k, v in confidence_scores.items()},
        "image_path": image_path,
        "fatigue_status": fatigue_status
    }

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)