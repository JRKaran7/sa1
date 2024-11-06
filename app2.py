import cv2
import numpy as np
import joblib
import mediapipe as mp
import pywhatkit as kit
import threading
import streamlit as st

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to extract keypoints from a frame
def extract_keypoints_from_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        keypoints = []
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i != 0:  # Exclude the nose landmark
                keypoints.append([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints).flatten()  # Flatten keypoints for model input
    return None

# Function to send WhatsApp alert (if fainting detected)
def send_whatsapp_alert(number, message):
    def alert():
        try:
            kit.sendwhatmsg_instantly(number, message)
            print("WhatsApp message sent successfully!")
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
    
    threading.Thread(target=alert).start()

# Function for real-time fainting detection using webcam feed
def real_time_fainting_detection(model):
    cap = cv2.VideoCapture(0)  # Using the default webcam (0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract the same points used during training
            points = [
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
            ]
        
            # Flatten the points and make predictions
            feature_vector = np.array(points).flatten().reshape(1, -1)
            prediction = model.predict(feature_vector)
            prediction_prob = model.predict_proba(feature_vector)

            # Show the prediction probabilities for faint, sitting, and standing
            prob_faint = prediction_prob[0][1] * 100
            prob_sitting = prediction_prob[0][2] * 100
            prob_standing = prediction_prob[0][3] * 100
        
            # Display probabilities on the frame
            text_faint = f'Faint Probability: {prob_faint:.2f}%'
            text_sitting = f'Sitting Probability: {prob_sitting:.2f}%'
            text_standing = f'Standing Probability: {prob_standing:.2f}%'

            cv2.putText(frame, text_faint, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text_sitting, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text_standing, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
            # Display the detected class with highest probability
            if prob_faint > 95:
                cv2.putText(frame, 'Faint Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                whatsapp_number = "+917204677562"
                alert_message = "Fainting detected! Please check immediately."
                send_whatsapp_alert(whatsapp_number, alert_message)

            elif prob_sitting > 95:
                cv2.putText(frame, 'Sitting Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            elif prob_standing > 95:
                cv2.putText(frame, 'Standing Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw landmarks on the frame for better understanding
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the output frame
        cv2.imshow('Faint, Sitting, and Standing Detection', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load pre-trained model
model = joblib.load('modelok.pkl')
st.write("Model loaded successfully!")

# Streamlit user interface
st.title("Real-Time Fainting Detection with Pose Estimation")
st.text("Using webcam for real-time detection.")

# Streamlit camera input
camera_input = st.camera_input("Use your webcam")

if camera_input:
    st.write("Processing webcam input...")
    real_time_fainting_detection(model)
