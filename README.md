## Drowsiness, Emotion & Distraction Detection System

This project is a real-time **driver/attention monitoring system** that uses a webcam to detect:

- **Drowsiness** (eyes closed / falling asleep)
- **Facial emotion** (e.g. Happy, Sad, Angry, Neutral, Tired)
- **Phone distraction** (using object detection)

When risky behavior is detected, the system triggers **audio alarms** and on-screen warnings to alert the user.

---

### Features

- **Real-time webcam monitoring**
  - Face and eye detection using OpenCV Haar Cascades
  - High-frequency frame processing with visual overlays

- **Drowsiness detection**
  - Eye state classification with a trained CNN (`eye_state_model_v3.h5`)
  - Triggers a loud **wake-up alarm** when eyes remain closed longer than a configurable threshold

- **Emotion recognition**
  - Facial emotion classification with a CNN (`emotion_model_v3.h5`)
  - Supports emotions such as: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, Tired
  - Live emotion label rendered on the video feed

- **Phone / distraction detection**
  - Object detection using **YOLOv8** (`yolov8n.pt`)
  - Detects **cell phones** and triggers a “put the phone away” alarm
  - Can be extended to other objects (e.g. laptop, book, pen)

- **Configurable audio alerts**
  - Custom alarm sounds for:
    - Drowsiness (`wake up speed.wav`)
    - Phone distraction (`danger but the phone away.wav`)
  - Optional selection of a custom alarm sound via a file dialog

- **Trainable models**
  - Scripts to:
    - Prepare datasets
    - Train emotion and drowsiness (eye state) models
    - Evaluate model performance

---

### Project Structure

The core project files are:

- **Runtime / main scripts**
  - `face_monitor.py`  
    Main entry point. Runs the full real-time monitoring system (drowsiness + emotion + phone detection + alarms).
  - `Face Emotion Alarm.py`  
    Alternative/legacy script focused mainly on emotion + alarm (optional).

- **Model training & data preparation**
  - `prepare_data.py`  
    Prepares face/emotion datasets (loading, preprocessing, splitting).
  - `prepare_sleep_data.py`  
    Prepares eye/drowsiness datasets.
  - `train_model.py`  
    Trains the **emotion recognition** model.
  - `train_sleep_model.py`  
    Trains the **eye state / drowsiness** model.
  - `test_model.py`  
    Evaluates trained models on test sets.

- **Models**
  - `emotion_model_v3.h5`  
    Latest emotion recognition model used at runtime.
  - `eye_state_model_v3.h5`  
    Latest eye state (open/closed) model used at runtime.
  - `yolov8n.pt`  
    YOLOv8 model for object/phone detection.

- **Data**
  - `data eyes/`  
    Raw/processed eye images for sleep/drowsiness detection.
  - `dataset/`  
    Raw/processed face images for emotion detection.
  - `test/`, `test images/`  
    Test data for models and visual validation.
  - `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`  
    Numpy arrays containing pre-split training and test datasets.

- **Audio**
  - `wake up speed.wav`  
    Alarm sound for drowsiness.
  - `danger but the phone away.wav`  
    Alarm sound for phone distraction.
  - `alert sound.mp3`  
    Additional alert sound (optional / for experiments).

---

### Requirements

The project was developed and tested with:

- **Python**: 3.10 (recommended)
- **Key Python packages**:
  - `numpy` (`< 2.0` for TensorFlow compatibility)
  - `tensorflow` (CPU version is sufficient)
  - `opencv-python`
  - `ultralytics` (for YOLOv8)
  - `pygame`
  - `pillow`
  - `tkinter` (usually included with Python on Windows)

A typical `requirements.txt` might look like:

```text
numpy<2
tensorflow<2.16
opencv-python
ultralytics
pygame
pillow
```

> **Note**  
> TensorFlow and NumPy versions must be compatible. If you encounter errors like “module compiled against NumPy 1.x cannot run on NumPy 2.x”, ensure you install `numpy<2`.

---

### Environment Setup

#### Option 1: Using Anaconda (recommended on Windows)

1. Open **Anaconda Prompt**.
2. Create and activate a dedicated environment:

```bash
conda create -n face-monitor python=3.10 -y
conda activate face-monitor
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

(or install individually)

```bash
pip install "numpy<2" "tensorflow<2.16" opencv-python ultralytics pygame pillow
```

#### Option 2: Using `venv` (standard Python)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### Running the Application

1. Open your terminal (Anaconda Prompt or activated `venv`).
2. Navigate to the project directory:

```bash
cd "C:\Users\lenovo\Downloads\Desktop\AI"
```

3. Ensure your webcam is connected and accessible.
4. Run the main script:

```bash
python face_monitor.py
```

5. A window titled **“Face Monitor”** should appear:
   - It will display your webcam feed.
   - Rectangles will appear around your face and eyes.
   - Text overlays will show:
     - Current **status**: Awake / WAKE UP!
     - **Eye open/closed** information and duration
     - **Detected emotion**
     - **Phone warning** if a phone is detected

6. Press **`q`** in the window to quit and release the camera.

---

### How It Works (High-Level)

- **Face & Eye Detection**
  - Uses OpenCV Haar Cascades to detect the face and eyes in each frame.
  - Focuses on the upper part of the face for more reliable eye detection.
  - Extracted eye regions are preprocessed (grayscale, resize, normalization) and fed into the eye state CNN.

- **Drowsiness Detection Logic**
  - The eye state model predicts the probability of the eyes being open.
  - When the system detects **closed eyes** for longer than a configured threshold (e.g. 1 second), it:
    - Marks the user as “sleeping”.
    - Triggers the drowsiness alarm sound in a loop until eyes reopen.

- **Emotion Recognition**
  - The face region is normalized and resized to the input size expected by the emotion model.
  - The emotion model predicts probabilities over predefined emotion classes.
  - The predicted emotion and confidence are rendered as overlay text.

- **Phone / Distraction Detection**
  - Frames are passed through YOLOv8 (`yolov8n.pt`).
  - If an object labeled **“cell phone”** (or similar) is detected with sufficient confidence:
    - A bounding box is drawn around the phone.
    - A “PUT THE PHONE AWAY!” message is displayed.
    - A phone-specific alarm sound is played until no phone is detected.

- **Alarms & Audio Handling**
  - `pygame.mixer` is used to load and loop alarm sounds.
  - Alarm playback is handled on separate threads to keep the video processing responsive.

---

### Training the Models

If you want to retrain or improve the models:

1. **Prepare data**
   - Place your raw images into the appropriate folders (e.g. `dataset/`, `data eyes/`, etc.).
   - Run:
     - `prepare_data.py` to generate datasets for emotion recognition.
     - `prepare_sleep_data.py` for eye state/drowsiness.

2. **Train models**
   - For emotion model:

     ```bash
     python train_model.py
     ```

   - For eye state / drowsiness model:

     ```bash
     python train_sleep_model.py
     ```

   - These scripts should output new `.h5` model files (e.g. `emotion_model_v3.h5`, `eye_state_model_v3.h5`).

3. **Test models**

   ```bash
   python test_model.py
   ```

4. **Update runtime models**
   - Ensure `face_monitor.py` is loading the latest model filenames.
   - Replace or archive older model versions as needed.

---

### Troubleshooting

- **`conda` is not recognized**
  - Make sure you are using **Anaconda Prompt**, not regular PowerShell or CMD.

- **NumPy / TensorFlow version errors**
  - Install compatible versions:

    ```bash
    pip install "numpy<2" "tensorflow<2.16"
    ```

- **Camera does not open**
  - Ensure no other application is using the webcam.
  - Try changing the camera index in `cv2.VideoCapture(0)` (e.g. `1` instead of `0`).

- **No sound / alarms**
  - Check that:
    - Audio files (`wake up speed.wav`, `danger but the phone away.wav`) exist in the expected directory.
    - System volume and `pygame.mixer` volume are not muted.

- **YOLO model not found / cannot download**
  - Ensure `yolov8n.pt` is present.
  - If ultralytics attempts to download but fails, check your internet connection or manually download the model and place it in the project folder.

---

### Future Improvements

- Add configuration file (e.g. `config.yaml`) for thresholds, paths, and device IDs.
- Refactor into a Python package structure (`src/`, `models/`, `data/`, `audio/`).
- Add a simple GUI to start/stop monitoring and switch modes (driver mode, study mode, etc.).
- Log events (drowsiness episodes, phone detections, emotions) for later analysis.

---

### License

Specify your license here (e.g. MIT, GPL, or “All rights reserved”), depending on how you intend this project to be used and shared.

