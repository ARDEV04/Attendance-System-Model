import face_recognition
import os
import pickle

DATASET_PATH = "dataset/"
MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "face_encodings.pkl")

def train_face_recognition():
    known_face_encodings = []
    known_face_names = []
    total_images = 0
    processed_images = 0

    # Ensure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    for student_folder in os.listdir(DATASET_PATH):
        student_path = os.path.join(DATASET_PATH, student_folder)

        if os.path.isdir(student_path):  # Ensure it's a folder
            for filename in os.listdir(student_path):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(student_path, filename)
                    total_images += 1

                    # Load the image
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image, num_jitters=5, model="large")  # More accurate encoding

                    if encodings:
                        known_face_encodings.append(encodings[0])  # Take the first detected face
                        known_face_names.append(student_folder)  # Use folder name as the student's name
                        processed_images += 1
                    else:
                        print(f"⚠️ No face found in: {image_path}")

    # Save trained face encodings
    if known_face_encodings:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print(f"✅ Training complete! Processed {processed_images}/{total_images} images. Model saved at {MODEL_PATH}")
    else:
        print("❌ No faces detected in dataset. Check images and try again.")

if __name__ == "__main__":
    train_face_recognition()
