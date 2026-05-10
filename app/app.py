import streamlit as st
import sounddevice as sd

from scipy.io.wavfile import write

import numpy as np
import pandas as pd

from datetime import datetime

import joblib

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

import plotly.express as px

import sys
import os

sys.path.append(
    os.path.abspath("..")
)

from utils.feature_extraction import extract_features

from chatbot import chatbot_response

from dashboard import show_dashboard

from face_emotion import detect_face_emotion

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(

    page_title="AI Emotion Recognition",

    page_icon="🧠",

    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""

<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: white;
    font-size: 50px;
    font-weight: bold;
}

.stButton>button {

    width: 100%;
    height: 3.5em;

    border-radius: 12px;

    background-color: #6C63FF;

    color: white;

    font-size: 18px;

    font-weight: bold;
}

[data-testid="metric-container"] {

    background-color: #1E1E1E;

    border-radius: 12px;

    padding: 15px;
}

</style>

""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = load_model(
    "../models/emotion_model.h5"
)

encoder = joblib.load(
    "../models/label_encoder.pkl"
)

# =========================
# HEADER
# =========================

st.title(
    "🧠 AI Multimodal Emotion Recognition"
)

st.write(
    "Real-Time Emotion Detection using Voice + Face"
)

# =========================
# LAYOUT
# =========================

left_col, right_col = st.columns([2,1])

# =========================
# LEFT SIDE
# =========================

with left_col:

    if st.button("🎤 Start Voice + Face Detection"):

        duration = 5
        fs = 22050

        # =====================
        # RECORD VOICE
        # =====================

        st.info("Recording voice...")

        recording = sd.rec(

            int(duration * fs),

            samplerate=fs,

            channels=1
        )

        sd.wait()

        write(
            "temp.wav",
            fs,
            recording
        )

        st.success(
            "Voice recording complete"
        )

        # =====================
        # FEATURE EXTRACTION
        # =====================

        feature = extract_features(
            "temp.wav"
        )

        # IMPORTANT FIX
        feature = pad_sequences(

            [feature],

            padding='post',

            dtype='float32'
        )

        # =====================
        # VOICE PREDICTION
        # =====================

        prediction = model.predict(
            feature
        )

        predicted_index = np.argmax(
            prediction
        )

        voice_emotion = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = np.max(
            prediction
        ) * 100

        # =====================
        # CONFIDENCE FILTER
        # =====================

        if confidence < 55:

            voice_emotion = "uncertain"

        # =====================
        # FACE DETECTION
        # =====================

        st.info(
            "Opening webcam... Press SPACE to capture face."
        )

        face_emotion = detect_face_emotion()

        # =====================
        # FUSION LOGIC
        # =====================

        if confidence < 60:

            final_emotion = face_emotion

        else:

            if face_emotion == voice_emotion:

                final_emotion = voice_emotion

            else:

                final_emotion = voice_emotion

        # =====================
        # METRICS
        # =====================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Voice Emotion",
            voice_emotion
        )

        col2.metric(
            "Face Emotion",
            face_emotion
        )

        col3.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.success(
            f"Final Emotion: {final_emotion}"
        )

        # =====================
        # PROBABILITY GRAPH
        # =====================

        st.subheader(
            "Emotion Probability Distribution"
        )

        emotions = encoder.classes_

        prob_df = pd.DataFrame({

            "Emotion": emotions,

            "Probability": prediction[0]
        })

        fig = px.bar(

            prob_df,

            x="Emotion",

            y="Probability",

            color="Emotion",

            title="Emotion Confidence Levels"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================
        # CHATBOT
        # =====================

        chatbot_reply = chatbot_response(
            final_emotion
        )

        st.info(
            f"🤖 Chatbot: {chatbot_reply}"
        )

        # =====================
        # SAVE HISTORY
        # =====================

        new_data = pd.DataFrame({

            "time": [datetime.now()],

            "emotion": [final_emotion]
        })

        new_data.to_csv(

            "../emotion_history.csv",

            mode='a',

            header=False,

            index=False,

            lineterminator='\n'
        )

        # =====================
        # DELETE TEMP AUDIO
        # =====================

        if os.path.exists("temp.wav"):

            os.remove("temp.wav")

# =========================
# RIGHT SIDE
# =========================

with right_col:

    st.subheader(
        "📊 System Information"
    )

    st.success(
        "CNN + LSTM Voice Recognition"
    )

    st.success(
        "DeepFace Face Recognition"
    )

    st.success(
        "Multimodal Emotion Fusion"
    )

    st.success(
        "Interactive Dashboard"
    )

# =========================
# DASHBOARD
# =========================

show_dashboard()