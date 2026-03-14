import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("emotion_model_v2.h5")
model.compile()  # Optional: Compile the model to avoid warnings

# Define emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to predict emotion from an image
def predict_emotion(image_path):
    # Load and convert image to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Resize to 48x48 (expected input size)
    img = cv2.resize(img, (48, 48))

    # Normalize and reshape for model input
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)  

    # Predict emotion
    predictions = model.predict(img)[0]
    emotion_index = np.argmax(predictions)
    emotion = emotions[emotion_index]
    confidence = predictions[emotion_index]

    print(f"Predicted Emotion: {emotion} ({confidence*100:.2f}%)")

# Test with an image
image_path = r"C:\Users\lenovo\Downloads\Desktop\AI\test images\Training_176862.jpg"
predict_emotion(image_path)
