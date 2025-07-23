class ConversationTrigger:
    def __init__(self, emotion: str, fatigue_status: str):
        """
        Initializes with emotion and fatigue_status.
        """
        self.emotion = emotion
        self.fatigue_status = fatigue_status

    def generate_message(self) -> str:
        """
        Applies rules and returns an appropriate first message.
        Returns None if no condition is met.
        """
        if self.fatigue_status == "fully fatigue":
            return "You seem very tired. Want to talk about what’s draining your energy?"
        elif self.emotion == "sad":
            return "It looks like you're feeling a bit down. I'm here for you. Want to share anything?"
        elif self.emotion == "angry":
            return "You seem frustrated. Do you want to talk about what's making you feel this way?"
        elif self.emotion == "fear":
            return "You're showing signs of stress. Is something on your mind?"
        elif self.emotion == "disgust":
            return "You look uncomfortable. Want to talk about it?"
        elif self.emotion == "happy":
            return "You seem happy! That’s wonderful. Want to share what made you smile?"
        return None
