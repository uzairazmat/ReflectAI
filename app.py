import streamlit as st
import cv2
import time
from datetime import datetime
import json

from emotion_detector.model import EmotionDetector
from emotion_detector.utils import save_image, log_to_file
from fatigue_detection import FatigueDetector

# ---------------- Streamlit UI Setup ----------------
st.set_page_config(page_title="Real-Time Emotion + Fatigue Detector", layout="centered")
st.title("🧠 ReflectAI: Real-Time Emotion + Fatigue Detector")

# Sidebar
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

# ---------------- Load Models ----------------
if "detector" not in st.session_state:
    st.session_state.detector = EmotionDetector()
detector = st.session_state.detector

if "fatigue" not in st.session_state:
    st.session_state.fatigue = FatigueDetector()
fatigue = st.session_state.fatigue

# ---------------- Init Variables ----------------
camera = cv2.VideoCapture(0)
frame_counter = 0
predicted_emotion = "neutral"
last_logged_time = time.time()
stable_count = 0
last_prediction = None

# ---------------- Real-Time Loop ----------------
while run:
    start_time = time.time()
    ret, frame = camera.read()
    if not ret:
        st.error("⚠️ Camera not accessible.")
        break

    frame = cv2.resize(frame, (640, 480))

    # 1. Fatigue Detection
    processed_frame, is_fatigued, ear = fatigue.process_frame(frame)

    # Determine fatigue status
    if ear is None:
        fatigue_status = "no face detected"
    elif ear >= fatigue.ear_threshold:
        fatigue_status = "not fatigue"
    elif fatigue.fatigue_counter >= fatigue.fatigue_frame_threshold:
        fatigue_status = "fully fatigue"
    else:
        fatigue_status = "normal fatigue"

    if is_fatigued:
        toast_placeholder.warning("😴 Fatigue Detected! Please rest.", icon="💤")

    # 2. Emotion Detection (runs every N frames)
    frame_counter += 1
    if frame_counter % frame_skip_rate == 0:
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        emotion, confidence_scores = detector.predict_emotion(rgb_frame)

        # Stability logic
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

                # Save data
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_path = save_image(frame, timestamp, emotion)

                # Store logs (both simple and detailed)
                log_to_file(
                    timestamp=timestamp,
                    emotion=emotion,
                    confidence_scores=confidence_scores,
                    image_path=image_path,
                    fatigue_status=fatigue_status
                )

                # UI Notification
                toast_placeholder.success(
                    f"✅ Saved: {emotion} | Fatigue: {fatigue_status}",
                    icon="💾"
                )

        # Show confidence scores
        confidence_placeholder.caption("💡 Confidence Scores:")
        confidence_placeholder.json({k: round(v, 2) for k, v in confidence_scores.items()})

    # 3. Show current emotion
    label_placeholder.markdown(f"### 😃 Current Emotion: `{predicted_emotion}`")

    # 4. Show delay time
    delay_placeholder.caption(f"🕒 Delay: `{round(time.time() - start_time, 2)}s`")

    # 5. Display processed frame
    cv2.putText(processed_frame, f"{predicted_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(display_frame)

    time.sleep(0.03)  # smooth stream

# In the else block (when webcam stops), replace the display code with:

else:
    camera.release()
    st.write("🛑 Webcam stopped.")

    # Display combined simple log with fatigue status
    st.subheader("📊 Emotion + Fatigue Log")
    simple_log = detector.get_predictions()  # Gets {"timestamp": "emotion"}

    # Load fatigue data from detailed logs
    try:
        with open("emotion_logs/detailed_predictions.json", "r") as f:
            detailed_logs = json.load(f)

        # Combine data for display
        combined_log = {}
        for ts, emotion in simple_log.items():
            # Find matching detailed entry (convert timestamp format if needed)
            file_ts = ts.replace(" ", "_").replace(":", "-")
            fatigue_status = detailed_logs.get(file_ts, {}).get("fatigue_status", "unknown")
            combined_log[ts] = f"{emotion}:{fatigue_status}"

        st.json(combined_log)

    except FileNotFoundError:
        st.json(simple_log)  # Fallback to original if no detailed logs
        st.warning("No fatigue data available")
    except json.JSONDecodeError:
        st.json(simple_log)
        st.error("Could not load fatigue data")

    # Detailed logs expander (unchanged)
    with st.expander("🔍 View Full Technical Logs"):
        try:
            with open("emotion_logs/detailed_predictions.json", "r") as f:
                st.json(json.load(f))
        except FileNotFoundError:
            st.warning("No detailed logs available")