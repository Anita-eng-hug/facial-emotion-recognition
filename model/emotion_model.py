
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Data directory containing FER2013 folders
data_dir = "fer2013/"
img_size = 48
batch_size = 64

# Augmentation + validation split
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, "train"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, "train"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
)

# Build CNN 
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
)

# ---- We save outputs in a consistent place ----
os.makedirs("model", exist_ok=True)

MODEL_OUT = os.path.join("model", "emotion_model.h5")
LABELS_OUT = os.path.join("model", "class_labels.json")

model.save(MODEL_OUT)
print(f"✅ Model saved to: {MODEL_OUT}")

# We save our class label mapping used by flow_from_directory
# Example: {"angry": 0, "disgust": 1, ...}
with open(LABELS_OUT, "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)

print("✅ Class indices saved to:", LABELS_OUT)
print("Class indices:", train_generator.class_indices)
