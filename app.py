import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('emotion_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Contempt']

# Preprocess frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    return np.expand_dims(resized, axis=(0, -1))

st.title("Real-Time Emotion Detection")
run = st.checkbox("Run Webcam")

if run:
    st.text("Press 'Stop' to exit.")
    cap = cv2.VideoCapture(0)  # Change `0` to the correct camera index if needed

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame)
        emotion = emotion_labels[np.argmax(predictions)]

        # Display prediction on frame
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f'Predicted Emotion: {emotion}')

    cap.release()
