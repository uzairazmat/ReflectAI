from llm_engine import LLMChatBot

model_path = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
log_path = "emotion_logs/emotions.json"

bot = LLMChatBot(model_path, log_path)

# First call triggers greeting based on log
print("Lumi:", bot.chat(""))

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = bot.chat(user_input)
    print("Lumi:", response)
