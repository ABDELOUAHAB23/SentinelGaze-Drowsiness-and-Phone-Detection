import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure dataset path is correct
dataset_path = os.path.join(os.getcwd(), "data eyes")  # Use full path

# Define image size and batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Image augmentation for better model generalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
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
