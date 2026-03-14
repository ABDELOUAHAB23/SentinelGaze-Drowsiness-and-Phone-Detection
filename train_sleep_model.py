import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Ensure dataset path is correct
dataset_path = os.path.join(os.getcwd(), "data eyes")  # Adjust if needed

# Define image size and batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Ensure dataset folders exist
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset path '{dataset_path}' does not exist!")

# Image augmentation for better model performance
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training dataset
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="binary",
    subset="training"
)

# Load validation dataset
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="binary",
    subset="validation"
)

# Print class indices to verify correct labels
print("Class Indices:", train_data.class_indices)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),  # 1 for grayscale
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (0 = Closed, 1 = Open)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_data,  # ✅ Ensure train_data is defined
    validation_data=val_data,
    epochs=30,  # Adjust based on performance
    verbose=1
)

# Save the trained model
model.save("eye_state_model_v3.h5")
print("Model saved successfully!")
