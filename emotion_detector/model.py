from deepface import DeepFace
from datetime import datetime
from collections import deque
import numpy as np
import time

class EmotionDetector:
    def __init__(self, smoothing_window=5):
        self.predictions = {}
        self.emotion_history = deque(maxlen=smoothing_window)
        self.last_logged_emotion = None

    def predict_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion_probs = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            self.emotion_history.append(emotion_probs)

            # Return smoothed emotion
            return self.get_smoothed_emotion(), emotion_probs

        except Exception as e:
            return "unknown", {}

    def get_smoothed_emotion(self):
        if not self.emotion_history:
            return "unknown"

        avg_emotions = {}
        for emotion in self.emotion_history[0]:
            avg_emotions[emotion] = np.mean([frame[emotion] for frame in self.emotion_history])

        return max(avg_emotions, key=avg_emotions.get)

    def log_emotion(self, emotion):
        if emotion != self.last_logged_emotion:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.predictions[timestamp] = emotion
            self.last_logged_emotion = emotion

    def get_predictions(self):
        return self.predictions
