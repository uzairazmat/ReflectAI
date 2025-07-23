import os
from typing import List, Dict
import random
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Gemini API package not found. Using mock responses.")


class LLMEngine:
    def __init__(self):
        """Initialize with psychology-informed response templates."""
        self.model = None
        self._initialize_llm()

        # Psychology-based response templates (CBT and Positive Psychology inspired)
        self.response_templates = {
            "happy": [
                "Glad you're feeling happy! Try savoring this moment. What's making you smile today?",
                "Happiness looks good on you! Want to share what's brightening your day?"
            ],
            "sad": [
                "This sadness is valid. Try the 5-4-3-2-1 grounding technique. What color do you see nearby?",
                "Hard days pass. Would a short walk outside help right now?"
            ],
            "angry": [
                "Anger is energy. Try box breathing (4-4-4-4). What needs to change here?",
                "Let's cool this down. Splash cold water on your wrists. What triggered this?"
            ],
            "tired": [
                "Your body needs care. Try 20-20-20: 20s stretch every 20 mins. Hydrated enough?",
                "Fatigue whispers before it shouts. Close your eyes for 30s. What's draining you?"
            ],
            "neutral": [
                "Let's tune in. How does your body feel right now - any tension or ease?",
                "Neutral is a good starting point. Want to explore what you're needing?"
            ]
        }

        # Fallback templates
        self.default_templates = [
            "Noticing {emotion}. Try {solution}. {question}",
            "With {fatigue_status} energy: {action}. {question}"
        ]

        # Evidence-based quick solutions
        self.solutions = {
            "happy": "share this feeling with someone",
            "sad": "name 3 things you see around you",
            "angry": "press palms together for 10 seconds",
            "tired": "drink some water and stretch up",
            "neutral": "take 2 conscious breaths"
        }

    def _initialize_llm(self):
        """Secure initialization of Gemini model."""
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("API key not found in environment variables")

                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    'models/gemini-2.0-flash',
                    generation_config={
                        "temperature": 0.5,  # More focused responses
                        "max_output_tokens": 50,  # Force brevity
                        "top_p": 0.9
                    },
                    safety_settings={
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                )
            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")
                self.model = None

    def generate_response(self, user_message: str, emotion: str = "neutral",
                          fatigue_status: str = "unknown",
                          conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate a 1–2 line psychology-informed response using CBT format.
        Uses templates first, then falls back to Gemini if available.
        """
        emotion = emotion.lower()

        # 1. Try psychology templates first (only if no conversation history provided)
        if not conversation_history and emotion in self.response_templates:
            return random.choice(self.response_templates[emotion])

        # 2. If Gemini model is available, use it
        if self.model and GEMINI_AVAILABLE:
            try:
                contents = []

                # 2.1. Add all previous messages to Gemini format
                if conversation_history:
                    for msg in conversation_history:
                        contents.append({
                            "role": msg["role"],  # 'user' or 'model'
                            "parts": [{"text": msg["content"]}]
                        })

                # 2.2. Add current message with extra context
                contents.append({
                    "role": "user",
                    "parts": [{
                        "text": f"""You are ReflectAI, a supportive and psychologically-informed chatbot trained in CBT and Positive Psychology.

    Emotion: {emotion}
    Fatigue status: {fatigue_status}

    Now respond to this message: "{user_message}"

    Your response should:
    - Be 1–2 short lines only
    - Use CBT/positive psychology techniques
    - Validate the emotion
    - Give an actionable tip or reflective question"""
                    }]
                })

                # ✅ LOG what is passed to Gemini (for debugging)
                st.write("🔍 Messages sent to Gemini:")
                st.json(contents)

                # 2.3. Generate response using Gemini model
                response = self.model.generate_content(contents)

                # 2.4. Trim long replies but avoid cutting mid-sentence
                return self._trim_response(response.text)

            except Exception as e:
                st.warning(f"Gemini generation failed. Using fallback response. Error: {str(e)}")

        # 3. Fallback: Use default template-based response
        template = random.choice(self.default_templates)

        fallback_data = {
            "emotion": emotion,
            "fatigue_status": fatigue_status,
            "solution": self.solutions.get(emotion, "pause and breathe"),
            "action": self.solutions.get(emotion, "pause and breathe"),
            "question": random.choice([
                "How does that sound?",
                "Want to try that?"
            ])
        }

        try:
            return template.format(**fallback_data)
        except KeyError:
            return "I'm here for you. Let's pause and breathe. Want to talk more?"

    def _trim_response(self, text: str) -> str:
        """Keep it concise but don’t cut mid-sentence."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        trimmed = ' '.join(lines[:2])  # Use 2 lines max
        return trimmed  # Don’t slice characters manually
