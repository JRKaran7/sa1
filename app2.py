import pywhatkit
import pyautogui
from pynput.mouse import Controller  # Importing Controller from pynput
import cv2
import mediapipe as mp
import numpy as np
import joblib
import streamlit as st

# Initialize the mouse controller
mouse = Controller()

# Define function to get current mouse position using pynput
def get_mouse_position():
    return mouse.position

# Example of using pyautogui and pywhatkit (Modify according to your need)
# Let's assume you're using pyautogui for some mouse actions
def perform_mouse_actions():
    # Move mouse to a specific position
    pyautogui.moveTo(100, 100)
    pyautogui.click()

# Initialize Mediapipe for pose detection (this is just a part of the original code)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Example of using OpenCV for capturing video frames
cap = cv2.VideoCapture(0)  # Using the default camera

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with pose landmarks
    cv2.imshow("Pose Detection", frame)

    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Get the current mouse position and print it
    current_position = get_mouse_position()
    print(f"Current Mouse Position: {current_position}")

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Example for saving a model using joblib (Modify the model as needed)
def save_model(model, filename="model.pkl"):
    joblib.dump(model, filename)

# Example for loading a saved model
def load_model(filename="modelok.pkl"):
    return joblib.load(filename)

# Streamlit section for the web application (if needed)
st.title("Pose Detection and Mouse Tracker")
st.write("This app tracks your mouse position and detects poses in real-time.")

# Use the model and display on the web
model = load_model("model.pkl")  # Assuming the model was saved earlier

# For Streamlit, display output in the app (Modify based on your use case)
st.write(f"Mouse Position: {get_mouse_position()}")
