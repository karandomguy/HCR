import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hcr_model.h5')

# Function to convert label index to character
def get_character(label):
    if label >= 0 and label <= 25:
        return chr(label + 65)
    return chr(label + 71)

# Initialize video capture object '0' stands for primary device
vid = cv2.VideoCapture(0)

WIDTH = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
# Check if the camera is opened
if not vid.isOpened():
    print("Error: Could not open video device.")
    exit()

# Capture and process the video frames
while True:
    ret, frame = vid.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (300, 300))
    normalized_frame = resized_frame / 255.0

    input_frame = np.expand_dims(normalized_frame, axis=0)
    cv2.imshow('test',input_frame[0])

    # Predict the character using the loaded model
    predictions = model.predict(input_frame)
    predicted_label = np.argmax(predictions[0])
    predicted_character = get_character(predicted_label)
    accuracy = predictions[0][predicted_label] * 100

    # Display the frame with predicted character and accuracy
    text = f"Character: {predicted_character}   Accuracy: {accuracy:.2f}%"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
vid.release()
cv2.destroyAllWindows()