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

        # Optimized thresholds for fatigue-aware detection
        self.thresholds = {
            'neutral': 0.65,  # High threshold reduces false neutrals
            'happy': 0.65,  # Increased to counter DeepFace happy bias
            'sad': 0.25,  # Lower for better fatigue correlation
            'angry': 0.35,  # Balanced for alert detection
            'fear': 0.30,  # Sensitive to stress indicators
            'surprise': 0.25,  # Brief reactions
            'disgust': 0.35  # Moderate threshold
        }

        # Emotion weights for fatigue context
        self.weights = {
            'sad': 1.2,  # Boost sadness relevance for fatigue
            'neutral': 0.9,  # Slightly reduce neutral
            'happy': 0.8  # Reduce happy bias
        }

    def predict_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'],
                                    enforce_detection=False, silent=True)
            emotion_probs = result[0]['emotion']

            # Apply weights and thresholds
            weighted_emotions = {
                e: (score / 100) * self.weights.get(e, 1.0)
                for e, score in emotion_probs.items()
                if (score / 100) > self.thresholds[e]
            }

            if weighted_emotions:
                dominant_emotion = max(weighted_emotions, key=weighted_emotions.get)
                self.emotion_history.append(emotion_probs)
                self.current_emotion = self._get_smoothed_emotion()
            else:
                # Fallback to highest probability above 80% of threshold
                fallback_emotions = {
                    e: score / 100
                    for e, score in emotion_probs.items()
                    if (score / 100) > (self.thresholds[e] * 0.8)
                }
                if fallback_emotions:
                    dominant_emotion = max(fallback_emotions, key=fallback_emotions.get)
                else:
                    dominant_emotion = self.current_emotion

            return dominant_emotion, emotion_probs

        except Exception as e:
            print(f"Emotion detection error: {str(e)}")
            return self.current_emotion, {}

    def _get_smoothed_emotion(self):
        """Weighted smoothing prioritizing fatigue-relevant emotions"""
        if not self.emotion_history:
            return "neutral"

        # Weighted average with fatigue priority
        avg_emotions = {}
        for emotion in self.emotion_history[0]:
            raw_scores = [frame[emotion] for frame in self.emotion_history]
            avg = np.mean(raw_scores) * self.weights.get(emotion, 1.0)
            avg_emotions[emotion] = avg

        return max(avg_emotions, key=avg_emotions.get)

    def log_emotion(self, emotion):
        """Only log when emotion changes"""
        if emotion != self.last_logged_emotion:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.predictions[timestamp] = emotion
            self.last_logged_emotion = emotion

    def get_predictions(self):
        """Returns simplified {timestamp: emotion} dict"""
        return self.predictions