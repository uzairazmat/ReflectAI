import streamlit as st
import cv2
import time
from datetime import datetime
from emotion_detector.model import EmotionDetector
from emotion_detector.utils import save_image, log_to_file

st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("🎭 Real-Time Emotion Detector")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
frame_skip_rate = st.sidebar.slider("Prediction Interval (every N frames)", 1, 30, 5)
stability_threshold = st.sidebar.slider("Stability Threshold (same emotion frames)", 1, 10, 3)
log_cooldown = st.sidebar.slider("Log Cooldown (seconds)", 1, 10, 3)

# UI Elements
run = st.checkbox("📸 Start Webcam")
FRAME_WINDOW = st.image([])
label_placeholder = st.empty()
delay_placeholder = st.empty()
confidence_placeholder = st.empty()
toast_placeholder = st.empty()

# Load Detector
if "detector" not in st.session_state:
    st.session_state.detector = EmotionDetector()
detector = st.session_state.detector

camera = cv2.VideoCapture(0)
frame_counter = 0
predicted_emotion = "neutral"
last_logged_time = time.time()
stable_count = 0
last_prediction = None

while run:
    start_time = time.time()
    ret, frame = camera.read()
    if not ret:
        st.error("⚠️ Camera not accessible.")
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_counter += 1
    if frame_counter % frame_skip_rate == 0:
        emotion, confidence_scores = detector.predict_emotion(rgb_frame)

        if emotion == last_prediction:
            stable_count += 1
        else:
            stable_count = 1
            last_prediction = emotion

        if stable_count >= stability_threshold and emotion != predicted_emotion:
            current_time = time.time()
            if current_time - last_logged_time >= log_cooldown:
                predicted_emotion = emotion
                detector.log_emotion(emotion)
                last_logged_time = current_time

                # Save image + log
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_path = save_image(frame, timestamp, emotion)
                log_to_file(timestamp, emotion, confidence_scores, image_path)

                # UI Notification
                toast_placeholder.success(f"✅ Saved image: {emotion} at {timestamp}", icon="💾")

        confidence_placeholder.caption("💡 Confidence Scores:")
        confidence_placeholder.json({k: round(v, 2) for k, v in confidence_scores.items()})

    label_placeholder.markdown(f"### 😃 Current Emotion: `{predicted_emotion}`")
    delay_placeholder.caption(f"🕒 Delay: `{round(time.time() - start_time, 2)}s`")

    cv2.putText(rgb_frame, f"{predicted_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    FRAME_WINDOW.image(rgb_frame)

    time.sleep(0.03)

else:
    camera.release()
    st.write("🛑 Webcam stopped.")
    st.subheader("📊 Emotion Log")
    st.json(detector.get_predictions())
