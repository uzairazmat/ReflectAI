import json
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
load_dotenv()




class ConversationManager:
    def __init__(self):
        # File paths
        self.current_session_file = "chat_llm/current_session_chat_history.json"
        self.chat_history_file = "chat_llm/chat_history.json"

        # Initialize files if they don't exist
        self._initialize_files()

        # Load current session history (empty at start)
        self.current_session_history = []

    def _initialize_files(self):
        """Ensure required files exist with proper structure."""
        os.makedirs("chat_llm", exist_ok=True)

        # Initialize current session file if not exists
        if not os.path.exists(self.current_session_file):
            with open(self.current_session_file, "w") as f:
                json.dump([], f)

        # Initialize chat history file if not exists
        if not os.path.exists(self.chat_history_file):
            with open(self.chat_history_file, "w") as f:
                json.dump([], f)

    def add_user_message(self, content: str):
        """Add a user message to current session history."""
        self.current_session_history.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save_current_session()

    def add_assistant_message(self, content: str):
        """Add an assistant message to current session history."""
        self.current_session_history.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save_current_session()

    def get_current_session_history(self):
        """Get all messages from current session."""
        return self.current_session_history

    def summarize_with_gemini(self, messages):
        """
        Summarizes a session's conversation using Gemini Flash API (Free Tier),
        considering emotional tone, mental health context, and personalization cues.
        """

        import os
        import requests

        # Load the free-tier Gemini Flash API key
        api_key = os.getenv("GEMINI_API_KEY_FOR_SUMMARY")
        if not api_key:
            raise ValueError("Gemini API key not found in environment variables.")

        # Convert the message list into a readable text format
        full_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
        )

        # Prompt designed for summarizing emotional wellness chat sessions
        prompt = (
            "You are a helpful assistant designed to summarize conversations from a personal emotional wellness assistant app.\n\n"
            "The summary should include:\n"
            "- The user's emotional state(s) throughout the conversation.\n"
            "- Any causes of distress, stress, joy, or frustration.\n"
            "- Useful advice or responses given by the assistant.\n"
            "- Repeating patterns or problems (e.g., stress, burnout, loneliness, frustration with tech, etc.).\n"
            "- Overall tone of the session and how the assistant responded.\n\n"
            "Be specific and concise. Avoid generic language.\n\n"
            "Conversation:\n\n"
            f"{full_text}"
        )

        # Gemini Flash endpoint for free-tier users
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
        }

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        # Send request to Gemini Flash API
        response = requests.post(
            f"{url}?key={api_key}",
            headers=headers,
            json=body,
        )

        # Extract the generated summary
        if response.status_code == 200:
            try:
                summary = response.json()['candidates'][0]['content']['parts'][0]['text']
                return summary
            except Exception as e:
                print("Error parsing Gemini response:", e)
                return None
        else:
            print("Gemini API Error:", response.text)
            return None

    def get_full_history_for_llm(self):
        """
        Get conversation history in format suitable for LLM.
        Returns: List of messages with role/content only
        """
        return [{"role": msg["role"], "content": msg["content"]}
                for msg in self.current_session_history]

    def _save_current_session(self):
        """Save current session to file (overwrite mode)."""
        with open(self.current_session_file, "w") as f:
            json.dump(self.current_session_history, f, indent=2)

    def save_to_chat_history(self):
        """
        Summarizes the current session and appends the summary to the global chat history.
        This helps in maintaining a long-term memory of user interactions in a lightweight format.
        """
        try:
            # Step 1: Load messages from current session
            if os.path.exists(self.current_session_file):
                with open(self.current_session_file, 'r') as file:
                    current_messages = json.load(file)
            else:
                print("No current session file found.")
                return

            # Step 2: Summarize the messages using Gemini API
            summary = self.summarize_with_gemini(current_messages)

            if not summary:
                print("Failed to generate summary.")
                return

            # Step 3: Create summary block with timestamp
            summary_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary
            }

            # Step 4: Load or create chat_history.json
            history = []
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r') as file:
                    try:
                        history = json.load(file)
                    except json.JSONDecodeError:
                        print("Corrupted history file, starting fresh.")

            # Step 5: Append summary and save back
            history.append(summary_entry)
            with open(self.chat_history_file, 'w') as file:
                json.dump(history, file, indent=2)

            print("✅ Session summary saved to chat history.")

        except Exception as e:
            print(f"❌ Error saving chat history: {e}")