import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#location of the dataset
data_dir = 'fer2013/'

# Set image size which has been given in the Kaggle discription
img_size = 48

# Settig the batch size
batch_size = 64

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Data generator for training set
train_generator = train_datagen.flow_from_directory(
    data_dir + 'train/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

# Data generator for validation set
validation_generator = train_datagen.flow_from_directory(
    data_dir + 'train/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Building the CNN model
model = Sequential([
    
    # First convolution layer — detects low-level features (edges, curves).
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),   # Reduces image size (down-sampling)

    # Second convolution layer — learns more detailed features.
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Third convolution layer — even deeper features.
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Fourth convolution layer — high-level features.
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Flatten converts 2D feature maps into a 1D vector.
    Flatten(),

    # Fully connected dense layer with 512 neurons.
    Dense(512, activation='relu'),

    # Dropout prevents overfitting by randomly dropping 50% of neurons.
    Dropout(0.5),

    # Output layer — 7 neurons for 7 emotion classes.
    Dense(7, activation='softmax')
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,         # Training images
    epochs=50,               # Train for 50 cycles
    validation_data=validation_generator   # Validation images
)

# Saves your trained model as a .h5 file.
model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")