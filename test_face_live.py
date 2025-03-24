import face_recognition
import pickle
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model  # For emotion detection

# Paths
MODEL_PATH = "models/face_encodings.pkl"
EMOTION_MODEL_PATH = "models/emotion_vgg16.h5"  # Update with your emotion model path
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def load_trained_models():
    """ Load the trained face recognition and emotion detection models. """
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Trained face model not found! Train the model first.")
        return None, None, None

    with open(MODEL_PATH, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    if not os.path.exists(EMOTION_MODEL_PATH):
        print("❌ Error: Emotion detection model not found!")
        return None, None, None

    emotion_model = load_model(EMOTION_MODEL_PATH)

    return known_face_encodings, known_face_names, emotion_model

def recognize_faces_and_emotions():
    """ Capture video from webcam and recognize faces and emotions in real-time. """
    known_face_encodings, known_face_names, emotion_model = load_trained_models()
    if known_face_encodings is None or emotion_model is None:
        return

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("🎥 Press 'q' to exit the webcam feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture image.")
            break

        # Convert frame from BGR (OpenCV) to RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
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

            # Preprocess for emotion model
            face_resized = cv2.resize(face_crop, (224, 224))  # Resize to 224x224 for model input
            face_resized = face_resized / 255.0  # Normalize
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension


            # Predict emotion
            emotion_prediction = emotion_model.predict(face_resized)
            emotion_index = np.argmax(emotion_prediction)
            emotion_label = EMOTION_LABELS[emotion_index]

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display name + emotion below the face
            display_text = f"{name} - {emotion_label}"
            cv2.putText(frame, display_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the output frame
        cv2.imshow("Face & Emotion Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_and_emotions()