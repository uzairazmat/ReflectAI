from deepface import DeepFace
from datetime import datetime
from collections import deque
import numpy as np



class EmotionDetector:
    def __init__(self, smoothing_window=5):
        self.predictions = {}
        self.emotion_history = deque(maxlen=smoothing_window)
        self.last_logged_emotion = None
        self.current_emotion = "neutral"

        # Updated thresholds (tuned for fatigue detection)
        self.thresholds = {
            'neutral': 0.85,  # Must be VERY certain to call something neutral
            'happy': 0.8,  # Requires intense smiles
            'sad': 0.4,  # Moderate (fatigue-relevant)
            'angry': 0.45,  # Clearly angry, not just tense
            'fear': 0.35,  # Only clear fear
            'surprise': 0.3,  # Brief, strong reactions
            'disgust': 0.4  # Explicit disgust
        }

    def predict_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            emotion_probs = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            confidence = emotion_probs[dominant_emotion] / 100

            # Only proceed if confidence > threshold
            threshold = self.thresholds.get(dominant_emotion, 0.5)  # Default: 50%
            if confidence > threshold:
                self.emotion_history.append(emotion_probs)
                smoothed_emotion = self._get_smoothed_emotion()
                self.current_emotion = smoothed_emotion
                return smoothed_emotion, emotion_probs
            else:
                # Return last valid emotion if threshold not met
                return self.current_emotion, emotion_probs

        except Exception:
            return self.current_emotion, {}

    def _get_smoothed_emotion(self):
        """Returns the dominant emotion from smoothed history."""
        if not self.emotion_history:
            return "neutral"

        # Average probabilities over the smoothing window
        avg_emotions = {
            emotion: np.mean([frame[emotion] for frame in self.emotion_history])
            for emotion in self.emotion_history[0]
        }
        return max(avg_emotions, key=avg_emotions.get)

    def log_emotion(self, emotion):
        if emotion != self.last_logged_emotion:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.predictions[timestamp] = emotion
            self.last_logged_emotion = emotion

    def get_predictions(self):
        return self.predictions