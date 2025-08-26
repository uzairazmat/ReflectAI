# ReflectAI

Reflect AI

Reflect AI is an experimental AI-powered tool designed to detect emotions, track fatigue levels, and provide therapeutic responses using Cognitive Behavioral Therapy (CBT) techniques.

The project is still in progress and is being actively developed into a more scalable and cost-effective MVP, migrating from Streamlit to Django.

🚀 Features

Emotion Detection via Webcam

Uses DeepFace to detect emotions from live video frames.

Takes N consecutive frames and updates the current emotion only if the same prediction appears N times (to ensure stability).

Fatigue Monitoring

Estimates user fatigue level from facial expressions and emotion patterns.

Rule-based Messaging

On the first detected emotion, Reflect AI sends a rule-based message.

Conversational Therapy

After user input, the system sends the following context to an LLM:

Previous history summary

Current detected emotion

Current session messages

The LLM then generates a therapeutic response inspired by CBT techniques.

Session Management

At the end of each session:

All messages from the session are summarized.

The summary is added to the previous history summary for continuity in future sessions.

🛠️ Tech Stack

Frontend / UI: Streamlit (migrating to Django MVP)

Emotion Detection: DeepFace

Machine Learning: LLM (for therapeutic response generation)

Backend: Python

Future Plans: Django-based scalable and cost-effective solution


📌 Current Status

✅ Basic working prototype in Streamlit

✅ Emotion detection and fatigue tracking

✅ LLM-based CBT response generation

🔄 Migrating from Streamlit to Django

🔄 Improving scalability, efficiency, and cost-effectiveness


⚠️ Disclaimer

Reflect AI is a work-in-progress research project.
It is not a replacement for professional mental health support. Please consult licensed professionals for medical or psychological help.
