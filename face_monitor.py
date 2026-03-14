import cv2
import numpy as np
import tensorflow as tf
import pygame
import time
import os
from PIL import Image
from threading import Thread
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from ultralytics import YOLO

class FaceMonitorSystem:
    def __init__(self):
        # Initialize face detection using OpenCV's Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize eye detection with more sensitive parameters
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Load emotion detection model
        self.emotion_model = self.load_emotion_model()
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Tired']
        
        # Load eye state model
        try:
            self.eye_state_model = tf.keras.models.load_model("eye_state_model_v3.h5")
            print("Loaded pre-trained eye state model")
        except Exception as e:
            print(f"Error loading eye state model: {e}")
            self.eye_state_model = None
        
        # Initialize video capture
        self.cap = None
        
        # Initialize alarm system with higher volume
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1.0)  # Set volume to maximum
        self.alarm_active = False
        self.phone_alarm_active = False
        self.alarm_thread = None
        self.phone_alarm_thread = None
        
        # Alarm sound files - store full paths
        self.wakeup_sound = os.path.join(os.path.dirname(__file__), "wake up speed.wav")
        self.phone_away_sound = os.path.join(os.path.dirname(__file__), "danger but the phone away.wav")
        
        # Verify audio files exist
        if not os.path.exists(self.wakeup_sound):
            print(f"Warning: Wake up sound file not found at {self.wakeup_sound}")
        if not os.path.exists(self.phone_away_sound):
            print(f"Warning: Phone away sound file not found at {self.phone_away_sound}")
        
        # Initialize state trackers with more sensitive sleep detection
        self.eye_closed_time = 0
        self.eye_closed_start_time = None
        self.sleep_threshold_time = 1.0  
        self.closed_eyes_frames = 0
        self.min_closed_frames = 2  
        self.phone_detected = False
        self.is_sleeping = False
        self.current_emotion = "Unknown"
        self.eyes_closed_start = None  # New: track when eyes first closed
        
        # Initialize YOLO model for object detection
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            print("Loaded YOLO model for phone detection")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

    def load_emotion_model(self):
        """Load or create an emotion detection model"""
        try:
            model = tf.keras.models.load_model("emotion_model_v3.h5")
            print("Loaded pre-trained emotion model")
            return model
        except:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(8, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("Created new emotion model (needs training)")
            return model

    def predict_eye_state(self, eye_img):
        """Predict if an eye is open or closed using the trained model"""
        try:
            if self.eye_state_model is None:
                return None
                
            # Preprocess the eye image
            eye_img = cv2.resize(eye_img, (24, 24))  # Resize to model input size
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            eye_img = eye_img / 255.0  # Normalize
            eye_img = eye_img.reshape(1, 24, 24, 1)  # Reshape for model input
            
            # Predict
            prediction = self.eye_state_model.predict(eye_img, verbose=0)[0]
            return prediction[0]  # Return probability of eye being open
            
        except Exception as e:
            print(f"Error predicting eye state: {e}")
            return None

    def detect_faces_and_eyes(self, frame):
        """Detect faces and eyes using OpenCV Haar Cascades and eye state model"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply image preprocessing for better detection
        gray = cv2.equalizeHist(gray)  # Improve contrast
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
        
        # Detect faces with more sensitive parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  
            minSize=(60, 60)  
        )
        
        if len(faces) == 0:  # No faces detected
            self.closed_eyes_frames = 0
            self.eyes_closed_start = None
            self.is_sleeping = False
            self.stop_alarm('sleep')
            return frame
            
        # Process the largest face (closest to camera)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Enhance face region for better eye detection
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # Calculate the upper half of the face for eye detection
        upper_face_roi = face_roi[int(h*0.1):int(h*0.5), :]  
        
        # Detect eyes in the upper face region with adjusted parameters
        eyes = self.eye_cascade.detectMultiScale(
            upper_face_roi,
            scaleFactor=1.05,  
            minNeighbors=3,    
            minSize=(25, 25),  
            maxSize=(w//3, h//3)  
        )
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Process each detected eye
        eyes_open = 0
        eye_confidences = []
        
        # Adjust eye coordinates relative to the upper face region
        y_offset = int(h*0.1)
        for (ex, ey, ew, eh) in eyes:
            # Adjust coordinates to full face
            ex_adj = ex
            ey_adj = ey + y_offset
            
            # Extract and preprocess eye region
            eye_roi = face_roi_color[ey_adj:ey_adj+eh, ex_adj:ex_adj+ew]
            
            if eye_roi.size == 0:  
                continue
                
            # Enhance eye region
            eye_roi_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
            
            eye_state = self.predict_eye_state(eye_roi)
            
            if eye_state is not None:
                eye_confidences.append(eye_state)
                if eye_state > 0.3:  
                    eyes_open += 1
                    color = (0, 255, 0)  # Green for open
                else:
                    color = (0, 0, 255)  # Red for closed
                
                # Draw rectangle around eye
                cv2.rectangle(frame, 
                            (x+ex_adj, y+ey_adj), 
                            (x+ex_adj+ew, y+ey_adj+eh), 
                            color, 2)
                
                # Show confidence
                cv2.putText(frame, f"{eye_state:.2f}", 
                          (x+ex_adj, y+ey_adj-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update sleep detection based on eye states
        current_time = time.time()
        
        # Consider eyes closed if no eyes detected or all detected eyes are closed
        eyes_closed = len(eyes) == 0 or (len(eye_confidences) > 0 and all(conf < 0.3 for conf in eye_confidences))
        
        if eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            
            # Calculate how long eyes have been closed
            eyes_closed_duration = current_time - self.eyes_closed_start
            
            # Update display with countdown
            remaining_time = max(0, self.sleep_threshold_time - eyes_closed_duration)
            cv2.putText(frame, f"Alert in: {remaining_time:.1f}s", (10, 150),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Trigger alarm after 1 second
            if eyes_closed_duration >= self.sleep_threshold_time:
                self.is_sleeping = True
                self.trigger_alarm('sleep')
        else:  # Eyes are open
            self.eyes_closed_start = None
            self.is_sleeping = False
            self.stop_alarm('sleep')
        
        # Add eye status text with duration
        status_text = f"Eyes: {eyes_open}/{len(eyes)} open"
        if self.eyes_closed_start is not None:
            duration = current_time - self.eyes_closed_start
            status_text += f" (Closed for {duration:.1f}s)"
        
        cv2.putText(frame, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 0) if not eyes_closed else (0, 0, 255), 2)
        
        # Add sleep status with warning
        if self.is_sleeping:
            sleep_status = "WAKE UP!"
            color = (0, 0, 255)  # Red
            thickness = 3
        else:
            sleep_status = "Awake"
            color = (0, 255, 0)  # Green
            thickness = 2
            
        cv2.putText(frame, f"Status: {sleep_status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
        
        # Detect emotion for the face
        try:
            face_img = cv2.resize(face_roi, (48, 48))
            emotion, _ = self.detect_emotion(cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR))
            self.current_emotion = emotion
            
            # Display emotion text
            cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error detecting emotion: {e}")
        
        return frame

    def detect_emotion(self, face_img):
        """Detect emotion from facial image"""
        try:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face / 255.0
            input_face = normalized_face.reshape(1, 48, 48, 1)
            
            emotion_probabilities = self.emotion_model.predict(input_face, verbose=0)
            emotion_index = np.argmax(emotion_probabilities[0])
            emotion = self.emotions[emotion_index]
            confidence = emotion_probabilities[0][emotion_index]
            
            return emotion, confidence
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown", 0.0

    def detect_phone(self, frame):
        """Detect phone in the frame using YOLO"""
        results = self.yolo_model(frame)
        
        phone_detected = False
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                cls_name = result.names[cls]
                conf = float(box.conf[0])
                
                # Check if the detected object is a cell phone/mobile phone with high confidence
                if (cls_name in ['cell phone', 'mobile phone'] and conf > 0.5):
                    phone_detected = True
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Phone ({conf:.2f})", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Always trigger alarm if phone is detected
        if phone_detected:
            if not self.phone_alarm_active:
                self.phone_detected = True
                self.trigger_alarm('phone')
        else:
            self.phone_detected = False
            if self.phone_alarm_active:
                self.stop_alarm('phone')
            
        return frame

    def detect_objects(self, frame):
        """Detect objects like cell phones, laptops, and books in the frame"""
        try:
            results = self.yolo_model(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.yolo_model.names[cls]
                    
                    if label in ["cell phone", "laptop", "book", "pen"]:  # Objects to detect
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error in object detection: {e}")
            
        return frame

    def trigger_alarm(self, alarm_type='sleep'):
        """Trigger the appropriate alarm based on the type"""
        if alarm_type == 'sleep' and not self.alarm_active:
            self.alarm_active = True
            if self.alarm_thread is None or not self.alarm_thread.is_alive():
                self.alarm_thread = Thread(target=self.play_alarm, args=(self.wakeup_sound, 'sleep'))
                self.alarm_thread.start()
        elif alarm_type == 'phone' and not self.phone_alarm_active:
            self.phone_alarm_active = True
            if self.phone_alarm_thread is None or not self.phone_alarm_thread.is_alive():
                self.phone_alarm_thread = Thread(target=self.play_alarm, args=(self.phone_away_sound, 'phone'))
                self.phone_alarm_thread.start()

    def stop_alarm(self, alarm_type='all'):
        """Stop the specified alarm(s)"""
        if alarm_type in ['sleep', 'all']:
            self.alarm_active = False
        if alarm_type in ['phone', 'all']:
            self.phone_alarm_active = False
        pygame.mixer.stop()

    def play_alarm(self, sound_file, alarm_type):
        """Play the specified alarm sound"""
        try:
            if not os.path.exists(sound_file):
                print(f"Error: Sound file not found: {sound_file}")
                return
                
            print(f"Playing {alarm_type} alarm: {sound_file}")  # Debug message
            
            # Load and play the new sound
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play(-1)  # -1 means loop indefinitely
            
            # Keep playing until alarm is stopped
            while (alarm_type == 'sleep' and self.alarm_active) or \
                  (alarm_type == 'phone' and self.phone_alarm_active):
                time.sleep(0.1)
            
            pygame.mixer.music.stop()
                
        except Exception as e:
            print(f"Error playing {alarm_type} alarm: {e}")

    def start_monitoring(self):
        """Start the face monitoring system"""
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process frame for faces and eyes
            processed_frame = self.detect_faces_and_eyes(frame)
            
            # Process frame for phone detection
            processed_frame = self.detect_phone(processed_frame)
            
            # Display status
            status_text = f"Status: {'WAKE UP!' if self.is_sleeping else 'Awake'}"
            cv2.putText(processed_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if self.is_sleeping else (0, 255, 0), 2 if self.is_sleeping else 1)
            
            if self.phone_detected:
                cv2.putText(processed_frame, "PUT THE PHONE AWAY!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow('Face Monitor', processed_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.stop_alarm('all')

    def select_alarm_sound(self):
        """Open a file dialog to select an alarm sound file"""
        self.root = tk.Tk()
        self.root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select Alarm Sound",
            filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav")]
        )
        if filename:
            self.wakeup_sound = filename
        self.root.destroy()

if __name__ == "__main__":
    # Create and run the face monitoring system
    system = FaceMonitorSystem()
    system.start_monitoring()