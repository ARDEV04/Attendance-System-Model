import face_recognition
import pickle
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = "models/face_encodings.pkl"
EMOTION_MODEL_PATH = "models/emotion_vgg16.h5"
DATA_FOLDER = "data"  # Folder to store attendance files

# Emotion Labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create "data" folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

# Function to get today's attendance file path
def get_attendance_file():
    today_date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(DATA_FOLDER, f"{today_date}.csv")

# Load trained models once at the start
def load_trained_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(EMOTION_MODEL_PATH):
        print("❌ Error: Required models not found! Train them first.")
        return None, None, None

    with open(MODEL_PATH, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    emotion_model = load_model(EMOTION_MODEL_PATH)

    return known_face_encodings, known_face_names, emotion_model

# Load today's attendance at the start
def load_attendance():
    file_path = get_attendance_file()
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame(columns=["Name", "Time", "Emotion"])

# Mark attendance in memory and save only when required
def mark_attendance(name, emotion, attendance_df):
    if (attendance_df["Name"] == name).any():
        return attendance_df  # If already marked, return as is

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    # Append new entry using pd.concat()
    new_entry = pd.DataFrame([{"Name": name, "Time": current_time, "Emotion": emotion}])
    attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)

    # Save to CSV only once
    attendance_df.to_csv(get_attendance_file(), index=False)
    print(f"✅ Marked Attendance: {name} - {emotion} at {current_time}")
    
    return attendance_df

# Main function
def recognize_faces_and_emotions():
    known_face_encodings, known_face_names, emotion_model = load_trained_models()
    if known_face_encodings is None or emotion_model is None:
        return

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("🎥 Press 'q' to exit the webcam feed.")
    
    attendance_df = load_attendance()  # Load attendance once

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture image.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

            # Extract face for emotion detection
            face_crop = frame[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            # Resize and preprocess for emotion model
            face_resized = cv2.resize(face_crop, (224, 224))
            face_resized = face_resized / 255.0  # Normalize
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

            # Predict emotion
            emotion_prediction = emotion_model.predict(face_resized, verbose=0)  # No logs for faster performance
            emotion_label = EMOTION_LABELS[np.argmax(emotion_prediction)]

            # Mark attendance in memory & save only if needed
            attendance_df = mark_attendance(name, emotion_label, attendance_df)

            # Display text
            display_text = f"{name} - {emotion_label} (Done ✅)" if name in attendance_df["Name"].values else f"{name} - {emotion_label} (Marked ✅)"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the output frame
        cv2.imshow("Face & Emotion Recognition - Attendance System", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_and_emotions()
