import cv2
import numpy as np
import streamlit as st
import pyautogui
import time
import operator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

categories = {
    0: 'palm',
    1: 'fist',
    2: 'thumbs-up',
    3: 'thumbs-down',
    4: 'index-right',
    5: 'index-left',
    6: 'no-gesture'
}

def main():
    st.markdown(
        "<h1 style='text-align:center;'>Hand Gesture Recognition Web App</h1>",
        unsafe_allow_html=True
    )

    pages = ['About Web App','Project Demo','Gesture Control Page']
    page = st.sidebar.selectbox('', pages)

    if page == 'About Web App':
        st.write("Control media player using hand gestures.")

    elif page == 'Project Demo':
        st.video("demo.mp4")

    elif page == 'Gesture Control Page':
        run = st.button('Start Web Camera')

        FRAME_WINDOW1 = st.image([])
        FRAME_WINDOW2 = st.image([])

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

if __name__ == "__main__":
    main()