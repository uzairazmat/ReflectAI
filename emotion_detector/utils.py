import os
import time
from datetime import datetime  # Add this if not already present
import cv2
import json

def save_image(frame, timestamp, emotion):
    os.makedirs("emotion_logs/images", exist_ok=True)
    path = f"emotion_logs/images/{timestamp}_{emotion}.jpg"
    cv2.imwrite(path, frame)
    return path


def log_to_file(timestamp, emotion, confidence_scores, image_path, fatigue_status="not detected"):
    os.makedirs("emotion_logs", exist_ok=True)
    log_path = "emotion_logs/detailed_predictions.json"

    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = {}

    data[timestamp] = {
        "emotion": emotion,
        "confidence": {k: round(float(v), 2) for k, v in confidence_scores.items()},
        "image_path": image_path,
        "fatigue_status": fatigue_status
    }

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)


def get_session_id():
    """Generates a unique session ID for each app run"""
    session_file = "emotion_logs/current_session.txt"

    # If session file exists, read the ID
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            return f.read().strip()

    # Create new session ID
    new_session_id = str(int(time.time()))  # Using timestamp as session ID
    os.makedirs("emotion_logs", exist_ok=True)
    with open(session_file, "w") as f:
        f.write(new_session_id)
    return new_session_id


def log_first_prediction(timestamp, emotion, confidence_scores, image_path, fatigue_status="not detected"):
    os.makedirs("emotion_logs", exist_ok=True)
    first_pred_path = "emotion_logs/first_predictions.json"

    # Check if we've already logged for this session
    session_id = get_session_id()
    session_file = f"emotion_logs/session_{session_id}.tmp"

    # If session marker exists, we've already logged first prediction
    if os.path.exists(session_file):
        return

    # Create session marker file
    with open(session_file, "w") as f:
        f.write("1")

    # Store the prediction data (overwrites completely)
    data = {
        "current_session": {
            "session_id": session_id,
            "timestamp": timestamp,
            "emotion": emotion,
            "confidence": {k: round(float(v), 2) for k, v in confidence_scores.items()},
            "image_path": image_path,
            "fatigue_status": fatigue_status,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    with open(first_pred_path, "w") as f:
        json.dump(data, f, indent=2)



def get_current_session_first_prediction():
    first_pred_path = "emotion_logs/first_predictions.json"
    session_id = get_session_id()

    if os.path.exists(first_pred_path):
        try:
            with open(first_pred_path, "r") as f:
                data = json.load(f)
                return data.get(session_id, None)
        except (json.JSONDecodeError, ValueError):
            return None
    return None