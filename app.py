
import os
import json

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# -----------------------------
# App + paths
# -----------------------------
app = Flask(__name__)

MODEL_PATH = os.path.join("model", "emotion_model.h5")  
LABELS_PATH = os.path.join("model", "class_labels.json")  
UPLOAD_FOLDER = os.path.join("static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load model once
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Load label order (index -> label)
# -----------------------------
def load_emotion_labels(labels_path: str):
    """
    Loads the exact class index mapping used during training.
    Expected JSON format from ImageDataGenerator.flow_from_directory:
      {"angry": 0, "disgust": 1, ...}

    Returns:
      list[str] where emotions[idx] gives the label for that output neuron.
    """
    if not os.path.exists(labels_path):
        # Fallback: You can still run, but results may be mislabeled.
        # It's better to generate class_labels.json from training (below).
        return ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    with open(labels_path, "r") as f:
        class_indices = json.load(f)

    emotions = [None] * len(class_indices)
    for name, idx in class_indices.items():
        emotions[int(idx)] = name.strip().title()

    return emotions


EMOTIONS = load_emotion_labels(LABELS_PATH)

DESCRIPTIONS = {
    "Happy": "Keep smiling! Share your joy today.",
    "Sad": "Don't worry, better days are ahead.",
    "Angry": "Take a deep breath and relax.",
    "Surprise": "Wow! Something unexpected!",
    "Neutral": "Calm and composed — keep going.",
    "Fear": "Stay strong. Courage conquers fear.",
    "Disgust": "Try shifting your focus to something positive.",
}

# -----------------------------
# Preprocessing 
# -----------------------------
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Matches your training setup:
    - grayscale
    - 48x48
    - rescale 1/255
    - shape: (1, 48, 48, 1) for Conv2D input
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((48, 48))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (48, 48)
    arr = np.expand_dims(arr, axis=-1)               # (48, 48, 1)
    arr = np.expand_dims(arr, axis=0)                # (1, 48, 48, 1)
    return arr


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    # Validate request
    if "image" not in request.files:
        return jsonify({"error": "No file field named 'image' found."}), 400

    file = request.files["image"]
    if not file or file.filename.strip() == "":
        return jsonify({"error": "No file selected."}), 400

    # Save upload
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess
    x = preprocess_image(filepath)

    # Predict
    probs = model.predict(x)
    idx = int(np.argmax(probs[0]))
    emotion = EMOTIONS[idx] if idx < len(EMOTIONS) else "Unknown"

    # Message
    description = DESCRIPTIONS.get(emotion, "")

    return jsonify({"emotion": emotion, "description": description})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
