import cv2
import numpy as np
import streamlit as st
import pyautogui
import time
import operator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import speech_recognition as sr
import threading

import whisper

# -------- SESSION STATE SAFE INIT --------
if "recording" not in st.session_state:
    st.session_state["recording"] = False

if "notes_text" not in st.session_state:
    st.session_state["notes_text"] = ""

if "summary" not in st.session_state:
    st.session_state["summary"] = ""

st.set_page_config(
    page_title="Gesture controlled lecture notes taking system using deep learning",
    page_icon="🖐️",
    layout="wide"
)

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
body {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1.5rem;
}

/* ---------- HEADERS ---------- */
.main-header {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-header {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

/* ---------- CARDS ---------- */
.card {
    background: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* ---------- GRADIENT HEADERS ---------- */
.gradient-purple {
    background: linear-gradient(90deg, #7b2ff7, #f107a3);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

.gradient-red {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}

/* ---------- BADGES ---------- */
.badge {
    background: #1f6feb;
    padding: 6px 12px;
    border-radius: 8px;
    margin: 4px;
    display: inline-block;
}

/* ---------- FEATURE GRID ---------- */
.feature-box {
    background: #161b22;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

loaded_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(120,120,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(7, activation='softmax')
])

loaded_model.load_weights("gesture-model.h5")

model = whisper.load_model("tiny")  # use "small" if laptop is good

categories = {
    0: 'palm',
    1: 'fist',
    2: 'thumbs-up',
    3: 'thumbs-down',
    4: 'index-right',
    5: 'index-left',
    6: 'no-gesture'
}

recognizer = sr.Recognizer()
notes_text = ""
recording = False

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time

def record_with_whisper():
    global notes_text, recording

    samplerate = 48000
    device_id = 1  # Stereo Mix (MME)

    while recording:
        try:
            # Record 5 seconds system audio
            audio_data = sd.rec(
                int(5 * samplerate),
                samplerate=samplerate,
                channels=1,
                device=device_id
            )
            sd.wait()

            # Convert to float32 (Whisper expects this)
            audio_data = audio_data.flatten().astype(np.float32)

            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data) + 1e-6)

            # 🔥 Directly pass audio (NO FILE)
            result = model.transcribe(audio_data, fp16=False)

            text = result["text"]

            if text.strip():
                print("Detected:", text)
                notes_text += text + "\n"

                with open("notes.txt", "a") as f:
                    f.write(text + "\n")

        except Exception as e:
            print("ERROR:", e)

#import sounddevice as sd
#print(sd.query_devices())

def main():
    st.markdown(
        "<h1 style='text-align:center;'>Hand Gesture Recognition Web App</h1>",
        unsafe_allow_html=True
    )

    pages = [
    '📘 About Web App',
    #'🎬 Project Demo',
    '🖐️ Gesture Control',
    '📝 Notes Taking'
    ]
    page = st.sidebar.selectbox('', pages)

    if page == '📘 About Web App':

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("### 👋 Gesture Control")
            st.markdown("""
            - Media control via hand gestures  
            - Real-time detection  
            - Volume & playback control  
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("### 🎤 Speech to Text")
            st.markdown("""
            - Auto note generation  
            - Whisper AI transcription  
            - Real-time updates  
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("### 🎥 Video Intelligence")
            st.markdown("""
            - Background video processing  
            - AI summarization  
            - Smart notes extraction  
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### ⚙️ Tech Stack")

        st.markdown("""
        <div class="badge">🤖 TensorFlow</div>
        <div class="badge">📷 OpenCV</div>
        <div class="badge">🌐 Streamlit</div>
        <div class="badge">🎙️ Whisper</div>
        """, unsafe_allow_html=True)

        # elif page == '🎬 Project Demo':

        #     st.markdown('<div class="card">', unsafe_allow_html=True)
        #     st.markdown("### 🎬 Demo Preview")
        #     st.video("demo.mp4")
        #     st.markdown('</div>', unsafe_allow_html=True)

    elif page == '🖐️ Gesture Control':
        st.markdown("### 🖐️ Gesture Control Panel")
        col1, col2 = st.columns([2,1])

        with col1:
            run = st.button('▶ Start Camera')

            FRAME_WINDOW1 = st.image([])
            FRAME_WINDOW2 = st.image([])

        with col2:
            st.markdown("### 📋 Gesture Legend")
            st.markdown("""
            - 🖐️ Palm → Play/Pause  
            - ✊ Fist → Mute  
            - 👍 Thumbs Up → Volume +  
            - 👎 Thumbs Down → Volume -  
            - ☝️ Index Right → Next  
            - ☝️ Index Left → Previous  
            """)
            
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

        while run:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (120,120))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, test_image = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

            FRAME_WINDOW1.image(test_image)

            result = loaded_model.predict(test_image.reshape(1,120,120,1), verbose=0)

            prediction = {
                'palm': result[0][0],
                'fist': result[0][1],
                'thumbs-up': result[0][2],
                'thumbs-down': result[0][3],
                'index-right': result[0][4],
                'index-left': result[0][5],
                'no-gesture': result[0][6]
            }

            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            label = prediction[0][0]
            action = "NO ACTION"

            if label == 'palm':
                action = "PLAY/PAUSE"
                pyautogui.press('playpause')
                time.sleep(0.5)
            elif label == 'fist':
                action = "MUTE"
                pyautogui.press('volumemute')
                time.sleep(0.5)
            elif label == 'thumbs-up':
                action = "VOLUME UP"
                pyautogui.press('volumeup')
            elif label == 'thumbs-down':
                action = "VOLUME DOWN"
                pyautogui.press('volumedown')
            elif label == 'index-right':
                action = "FORWARD"
                pyautogui.press('nexttrack')
            elif label == 'index-left':
                action = "REWIND"
                pyautogui.press('prevtrack')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.putText(frame, f"Gesture: {label}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            cv2.putText(frame, f"Action: {action}", (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            FRAME_WINDOW2.image(frame)

        camera.release()
        cv2.destroyAllWindows()
    
    elif page == '📝 Notes Taking':

        st.markdown('<div class="gradient-purple">🎤 Speech-to-Text Notes</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # ▶ START
        if col1.button("▶ Start Recording"):
            if not st.session_state.get("recording", False):
                st.session_state["recording"] = True   # ✅ FIXED
                st.session_state["notes_text"] = ""
                open("notes.txt", "w").close()

                thread = threading.Thread(target=record_with_whisper, daemon=True)
                thread.start()

        # ⏹ STOP
        if col2.button("⏹ Stop Recording"):
            st.session_state["recording"] = False   # ✅ FIXED

        # 🔴 STATUS
        if st.session_state.get("recording", False):
            st.success("🔴 Recording in progress...")
        else:
            st.info("⚪ Idle")

        # 📄 INFO
        st.info("📄 Notes are being saved automatically to 'notes.txt' in your project folder.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

if __name__ == "__main__":
    main()