import streamlit as st
import cv2
import time
from datetime import datetime
import json
import os
from chat_llm.SessionManager import SessionManager
from chat_llm.conversation_trigger import ConversationTrigger
from chat_llm.conversation_handler import ConversationManager
from chat_llm.llm_engine import LLMEngine

from emotion_detector.model import EmotionDetector
from emotion_detector.utils import save_image, log_to_file, log_first_prediction
from fatigue_detection import FatigueDetector
from emotion_detector.utils import get_session_id, log_first_prediction

# ---------------- Streamlit UI Setup ----------------
st.set_page_config(page_title="🧠 ReflectAI", layout="centered")
st.title("🧠 ReflectAI: Real-Time Emotion + Fatigue Detector")

# ============== Chat Initialization ==============
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

if "llm_engine" not in st.session_state:
    st.session_state.llm_engine = LLMEngine()

if "message_shown" not in st.session_state:
    st.session_state.message_shown = False

if "first_prediction_logged" not in st.session_state:
    st.session_state.first_prediction_logged = False

# NEW flag: has a session actually started?
if "session_started" not in st.session_state:
    st.session_state.session_started = False

# Clear previous session file
session_file = "emotion_logs/current_session.txt"
if os.path.exists(session_file):
    os.remove(session_file)

# ============== Sidebar Settings ==============
st.sidebar.header("⚙️ Settings")
frame_skip_rate     = st.sidebar.slider("Prediction Interval (every N frames)", 1, 30, 5)
stability_threshold = st.sidebar.slider("Stability Threshold (same emotion frames)", 1, 10, 3)
log_cooldown        = st.sidebar.slider("Log Cooldown (seconds)", 1, 10, 3)

# Start/Stop Webcam
run = st.checkbox("📸 Start Webcam")

# If user just checked it, mark session as started
if run and not st.session_state.session_started:
    st.session_state.session_started = True

# ============== Chat Interface ==============
FRAME_WINDOW        = st.image([])
label_placeholder   = st.empty()
delay_placeholder   = st.empty()
confidence_placeholder = st.empty()
toast_placeholder   = st.empty()

st.markdown("---")
st.markdown("## 💬 Conversation")
for message in st.session_state.conversation_manager.get_current_session_history():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.conversation_manager.add_user_message(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_history = st.session_state.conversation_manager.get_full_history_for_llm()
            response = st.session_state.llm_engine.generate_response(
                user_message=user_input,
                conversation_history=full_history
            )
            st.markdown(response)
            st.session_state.conversation_manager.add_assistant_message(response)

# ---------------- Load Emotion/Fatigue Models ----------------
if "detector" not in st.session_state:
    st.session_state.detector = EmotionDetector()
detector = st.session_state.detector

if "fatigue" not in st.session_state:
    st.session_state.fatigue = FatigueDetector()
fatigue = st.session_state.fatigue

# ---------------- Camera Setup ----------------
if "camera" not in st.session_state:
    try:
        st.session_state.camera = cv2.VideoCapture(0)
        if not st.session_state.camera.isOpened():
            st.error("⚠️ Failed to initialize camera")
            del st.session_state.camera
    except Exception as e:
        st.error(f"⚠️ Camera init error: {e}")
        if "camera" in st.session_state:
            del st.session_state.camera

camera = st.session_state.get("camera", None)
if camera is None and run:
    st.error("Camera not available. Please refresh.")
    st.stop()

# ---------------- Tracking Variables ----------------
frame_counter      = 0
predicted_emotion  = "neutral"
last_logged_time   = time.time()
stable_count       = 0
last_prediction    = None

# ---------------- Main Loop ----------------
if run and camera:
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("⚠️ Camera not accessible.")
            break

        start_time = time.time()
        frame = cv2.resize(frame, (640, 480))

        # Fatigue detection
        processed_frame, is_fatigued, ear = fatigue.process_frame(frame)
        if ear is None:
            fatigue_status = "no face detected"
        elif ear >= fatigue.ear_threshold:
            fatigue_status = "not fatigue"
        elif fatigue.fatigue_counter >= fatigue.fatigue_frame_threshold:
            fatigue_status = "fully fatigue"
        else:
            fatigue_status = "normal fatigue"

        if is_fatigued:
            toast_placeholder.warning("😴 Fatigue Detected! Please rest.")

        # Emotion detection
        frame_counter += 1
        if frame_counter % frame_skip_rate == 0:
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            emotion, confidence_scores = detector.predict_emotion(rgb_frame)

            if emotion == last_prediction:
                stable_count += 1
            else:
                stable_count = 1
                last_prediction = emotion

            if stable_count >= stability_threshold and emotion != predicted_emotion:
                if time.time() - last_logged_time >= log_cooldown:
                    predicted_emotion = emotion
                    last_logged_time = time.time()
                    detector.log_emotion(emotion)

                    # Save image & logs
                    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    image_path = save_image(frame, timestamp, emotion)
                    log_to_file(timestamp, emotion, confidence_scores, image_path, fatigue_status)
                    if not st.session_state.first_prediction_logged:
                        log_first_prediction(timestamp, emotion, confidence_scores, image_path, fatigue_status)
                        st.session_state.first_prediction_logged = True

                    # Rule-based first message
                    session_id = get_session_id()
                    session_manager = SessionManager()
                    if session_manager.is_current_session(session_id) and not st.session_state.message_shown:
                        emo, fat = session_manager.get_emotion_and_fatigue()
                        if emo and fat:
                            trigger = ConversationTrigger(emo, fat)
                            first_msg = trigger.generate_message()
                            if first_msg:
                                st.markdown("---")
                                st.markdown("### 🤖 ReflectAI wants to chat:")
                                with st.chat_message("assistant"):
                                    st.markdown(first_msg)
                                    st.session_state.conversation_manager.add_assistant_message(first_msg)
                                st.markdown("---")
                                st.session_state.message_shown = True

            confidence_placeholder.caption("💡 Confidence Scores:")
            confidence_placeholder.json({k: round(v,2) for k,v in confidence_scores.items()})

        # UI updates
        label_placeholder.markdown(f"### 😃 Current Emotion: `{predicted_emotion}`")
        delay_placeholder.caption(f"🕒 Delay: `{round(time.time()-start_time,2)}s`")
        cv2.putText(processed_frame, predicted_emotion, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        time.sleep(0.03)

# ---------------- Session End Logic ----------------
if not run and st.session_state.session_started:
    # Cleanup camera
    if camera:
        camera.release()
        del st.session_state.camera

    st.write("🛑 Webcam stopped.")

    # Save chat summary now
    try:
        st.session_state.conversation_manager.save_to_chat_history()
        st.success("✅ Session summary saved to chat history.")
    except Exception as e:
        st.error(f"⚠️ Failed to save conversation summary: {e}")

    # Display logs
    st.subheader("📊 Emotion + Fatigue Log")
    simple_log = detector.get_predictions()
    try:
        with open("emotion_logs/detailed_predictions.json","r") as f:
            detailed_logs = json.load(f)
        combined = {
            ts: f"{emo}:{detailed_logs.get(ts.replace(' ','_').replace(':','-'),{}).get('fatigue_status','unknown')}"
            for ts, emo in simple_log.items()
        }
        st.json(combined)
    except Exception:
        st.json(simple_log)

    with st.expander("🔍 View Full Technical Logs"):
        try:
            with open("emotion_logs/detailed_predictions.json","r") as f:
                st.json(json.load(f))
        except FileNotFoundError:
            st.warning("No detailed logs available")

    # Reset flags for next session
    st.session_state.session_started = False
    if "first_prediction_logged" in st.session_state:
        del st.session_state.first_prediction_logged
    if "message_shown" in st.session_state:
        del st.session_state.message_shown
