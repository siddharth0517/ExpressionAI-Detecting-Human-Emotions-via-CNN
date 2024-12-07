import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('emotion_model.keras')

# Map emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Contempt']

# Function to preprocess image
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    return np.expand_dims(resized, axis=(0, -1))  # Add batch and channel dimensions

# Streamlit UI
st.title("Real-Time Emotion Detection")
run = st.checkbox("Run Webcam")

if run:
    st.text("Webcam is running. Press 'Stop' to exit.")
    cap = cv2.VideoCapture(0)  # Access the webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Predict emotion
        predictions = model.predict(processed_frame)
        emotion = emotion_labels[np.argmax(predictions)]
        
        # Display the prediction on the video feed
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Show video feed in Streamlit
        st.image(frame)

    cap.release()
