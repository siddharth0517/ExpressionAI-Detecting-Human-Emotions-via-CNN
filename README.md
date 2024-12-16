# ExpressionAI-Detecting-Human-Emotions-via-CNN
This project leverages a **Convolutional Neural Network (CNN)** to detect human emotions in real-time using a webcam. The model is trained on the **FER2013+ dataset**, which consists of grayscale images of faces classified into 8 emotion categories.

## Emotion Categories
The model predicts the following emotions:

+ Angry
+ Disgust
+ Fear
+ Happy
+ Neutral
+ Sad
+ Surprise
+ Contempt

## Dataset
The dataset used for training the model is the FER2013+ dataset, which contains labeled grayscale images of size 48x48 pixels.

## Features
+ Real-time emotion detection via a webcam.
+ Preprocessing: The model preprocesses input frames by converting them to grayscale, resizing them to 48x48 pixels, and normalizing pixel values.
+ Deep Learning Model: Built using TensorFlow and Keras, with multiple convolutional layers to achieve high accuracy in emotion classification.

## Model Architecture
The CNN model consists of:

+ Convolutional Layers with ReLU activation.
+ Max Pooling Layers for dimensionality reduction.
+ Fully Connected Layers with Dropout for regularization.
+ Output Layer with Softmax activation for multi-class classification.

## Installation and Setup
### Prerequisites
Ensure you have the following installed:

+ Python 3.7 or higher
+ TensorFlow
+ OpenCV
+ NumPy

## How It Works
+ Capture Frames: The script captures video frames from your webcam.
+ Preprocess Frames: Each frame is converted to grayscale, resized to 48x48 pixels, and normalized.
+ Predict Emotion: The preprocessed frame is passed through the trained CNN model to predict the emotion.
+ Display Results: The predicted emotion is displayed on the video feed in real time.

### Example Output
When the model detects an emotion, it overlays the predicted label (e.g., Happy, Sad) on the webcam feed.
![happy](https://github.com/user-attachments/assets/585466b6-7133-4fd2-afd4-46299fd5d288)



## Limitations
+ The model's accuracy depends on the lighting conditions and the quality of the webcam feed.
+ Performance may degrade with non-frontal or obscured faces.

## Future Enhancements
+ Integration with more robust face detection algorithms.
+ Support for multi-face emotion detection in a single frame.
+ Deployment as a web or mobile application.
