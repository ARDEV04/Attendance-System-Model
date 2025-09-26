# Attendance-System-Model

Face-recognition–based attendance pipeline with simple training, live testing, and automated CSV logging.

> Repo layout (as of latest commit): `data/`, `dataset/`, `models/`, `attendance_marking.py`, `train_face.py`, `test_face_live.py`, `requirements.txt`, plus `.gitignore` and `.gitattributes`.

## ✨ Features

- Train a face model from your own image dataset.
- Live webcam test to verify recognition.
- One-click attendance run: recognize faces and save CSV logs (date/time stamped) to `data/`.
- Simple folder-based dataset management.

## 🗂️ Project Structure

```
.
├─ dataset/            # Training images; typically one folder per person
├─ models/             # Saved embeddings/classifier files after training
├─ data/               # Attendance CSVs and run outputs
├─ train_face.py       # Train model on images in ./dataset
├─ test_face_live.py   # Quick live webcam verification
├─ attendance_marking.py # Recognize + write attendance CSVs in ./data
├─ requirements.txt    # Python dependencies
├─ .gitignore, .gitattributes
```

## 🚀 Quickstart

### 1) Set up Python

- Use Python **3.9–3.11** (recommended).
- Create and activate a virtual env:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  ```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Prepare the dataset

Organize `./dataset` as **one folder per person**, each containing 10–30 clear images:

```
dataset/
  person1/
    img1.jpg
    img2.jpg
  person2/
    pic1.png
    pic2.png
```

### 4) Train the model

```bash
python train_face.py   --dataset ./dataset   --models ./models
```

### 5) Verify with live test (optional)

```bash
python test_face_live.py --source 0 --models ./models
```

### 6) Run attendance

```bash
python attendance_marking.py   --source 0   --models ./models   --out ./data
```

## 🛠️ Command-line Options (common)

Most scripts accept these flags (names may vary slightly; check `--help`):

- `--dataset PATH` : path to training images (default: `./dataset`)
- `--models PATH`  : path to save/load model files (default: `./models`)
- `--source N|PATH`: webcam index (e.g., `0`) or video file path
- `--out PATH`     : output directory for CSVs/logs (default: `./data`)

Example:
```bash
python attendance_marking.py --source 0 --models ./models --out ./data
```

## 📦 Outputs

- **Models:** saved into `./models` after training (encodings/classifier files).
- **Attendance CSVs:** saved into `./data` during runs; one row per recognized person with timestamp.

## ✅ Tips for Best Results

- Ensure your webcam is 720p or better and the scene is well lit.
- Train with 10–30 diverse images per person.
- Avoid heavy motion and extreme angles during attendance capture.
- If a person isn’t recognized, add more varied images for them and retrain.

## 🧪 Troubleshooting

- **No camera found:** try `--source 1` or a video file path; ensure permissions granted.
- **Low accuracy:** add more/better training images; ensure faces are large and clear.
- **Build errors on install:** install OS-specific build tools (C/C++ build tools on Windows; `cmake`, `build-essential`, and Python headers on Linux/macOS).

## 📁 Ignored Files / Large Assets

- Datasets, models, and generated CSVs are typically large/binary and may be git-ignored (see `.gitignore`).


- Code and structure by the repository author(s).  
- Computer-vision pipeline built on standard Python CV/ML libraries.
