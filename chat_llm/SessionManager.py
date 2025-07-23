import json
from typing import Optional

class SessionManager:
    def __init__(self, prediction_path: str = "emotion_logs/first_predictions.json"):
        self.prediction_path = prediction_path
        self.session_data = self._load_session_data()

    def _load_session_data(self):
        try:
            with open(self.prediction_path, "r") as f:
                data = json.load(f)
                return data.get("current_session")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def is_current_session(self, session_id: str) -> bool:
        """Check if given session_id matches the current session."""
        if not self.session_data:
            return False
        return self.session_data.get("session_id") == session_id

    def get_emotion_and_fatigue(self):
        """Return current emotion and fatigue status."""
        if not self.session_data:
            return None, None
        return self.session_data.get("emotion"), self.session_data.get("fatigue_status")
