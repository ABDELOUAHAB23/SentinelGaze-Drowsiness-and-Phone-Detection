import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define dataset paths
train_path = r"C:\Users\lenovo\Downloads\Desktop\AI\dataset\train"
test_path = r"C:\Users\lenovo\Downloads\Desktop\AI\dataset\test"

# Define emotion categories
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotions)

# Function to load images
def load_data(dataset_path):
    X, y = [], []
    for label, emotion in enumerate(emotions):
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️ Warning: {emotion} folder not found. Skipping...")
            continue
        for image_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, image_file)
            img = load_img(img_path, color_mode="grayscale", target_size=(48, 48))
            img_array = img_to_array(img) / 255.0  # Normalize
            X.append(img_array)
            y.append(label)
    return np.array(X).reshape(-1, 48, 48, 1), to_categorical(np.array(y), num_classes=num_classes)

# Load train and test datasets
print("📂 Loading Training Data...")
X_train, y_train = load_data(train_path)
print(f"✅ Loaded {len(X_train)} training images.")

print("📂 Loading Test Data...")
X_test, y_test = load_data(test_path)
print(f"✅ Loaded {len(X_test)} test images.")

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Dataset prepared and saved!")
