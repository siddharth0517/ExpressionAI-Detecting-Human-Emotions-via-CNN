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

# Streamlit app
st.title("Real-Time Emotion Detection")
st.text("Click the 'Start Webcam' button to begin.")

# Create a placeholder for the video feed
video_feed = st.empty()

# Add buttons for controlling the webcam
start_webcam = st.button("Start Webcam")
stop_webcam = st.button("Stop Webcam")

# Start webcam feed
if start_webcam:
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    st.text("Press 'Stop Webcam' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        # Process the frame for emotion detection
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame)
        emotion = emotion_labels[np.argmax(predictions)]

        # Overlay emotion prediction on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert frame from BGR to RGB (Streamlit uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the placeholder
        video_feed.image(frame_rgb, caption=f"Predicted Emotion: {emotion}", use_column_width=True)

        # Check if the stop button was pressed
        if stop_webcam:
            st.text("Webcam stopped.")
            break

    cap.release()
