import os

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

# 1) Create the Flask app instance
app = Flask(__name__)

# 2) Define where uploads will be stored (useful for debugging / optional caching)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the folder exists

# 3) Load the trained model once at startup (so every request doesn't reload it)
MODEL_PATH = os.path.join("model", "emotion_model.h5")
model = load_model(MODEL_PATH)


# Labels + user-facing text


# NOTE: The order here must match our model's output neuron order
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Short supportive messages associated with each predicted class
EMOTION_MESSAGES = {
    "Happy": "Keep smiling! Share your joy today.",
    "Sad": "Don't worry, better days are ahead.",
    "Angry": "Take a deep breath and relax.",
    "Surprise": "Wow! Something unexpected!",
    "Neutral": "Calm and composed — keep going.",
    "Fear": "Stay strong. Courage conquers fear.",
    "Disgust": "Try shifting your focus to something positive.",
}


# Image preprocessing

def prepare_image_for_model(image_path: str) -> np.ndarray:
    # 1) Load as grayscale (matches training)
    img = Image.open(image_path).convert("L")

    # 2) Resize to 48x48 (matches training)
    img = img.resize((48, 48))

    # 3) Convert to float + rescale (matches ImageDataGenerator(rescale=1./255))
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (48, 48)

    # 4) Add channel dim -> (48, 48, 1)
    arr = np.expand_dims(arr, axis=-1)

    # 5) Add batch dim -> (1, 48, 48, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr

#Declaring our routes and logic

@app.get("/")
def home():
    """Render our main UI page."""
    return render_template("index.html")


@app.post("/predict")
def predict_emotion():
    """
    Receive an uploaded image and return:
    - predicted emotion label
    - a short description/message
    """
    # Step 1: Validate that the request contains a file
    if "image" not in request.files:
        return jsonify({"error": "No file field named 'image' found in request."}), 400

    uploaded = request.files["image"]
    if not uploaded or uploaded.filename.strip() == "":
        return jsonify({"error": "No file selected."}), 400

    # Step 2: Save the upload locally (optional, but keeps behavior identical)
    save_path = os.path.join(UPLOAD_DIR, uploaded.filename)
    uploaded.save(save_path)

    # Step 3: Preprocess for inference
    model_input = prepare_image_for_model(save_path)

    # Step 4: Run our inference
    probs = model.predict(model_input)

    # Step 5: Convert probabilities -> label (highest score wins)
    best_index = int(np.argmax(probs, axis=1)[0] if probs.ndim > 1 else np.argmax(probs))
    label = EMOTION_LABELS[best_index]

    # Step 6: Attach a friendly message
    message = EMOTION_MESSAGES.get(label, "")

    # Step 7: Respond as JSON
    return jsonify({"emotion": label, "description": message})



if __name__ == "__main__":
    # Vercel normally runs via its own entrypoint; this is for our local testing.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

