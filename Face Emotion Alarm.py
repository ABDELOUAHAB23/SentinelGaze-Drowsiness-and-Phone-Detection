import cv2
import numpy as np
import dlib
import time
import pygame
import mediapipe as mp
from keras.models import load_model

# Load pre-trained face and emotion models
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')  # Load a pre-trained emotion model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model

# Initialize alarm sound
pygame.mixer.init()
pilarm = "alarm_sound.mp3"  # Replace with an actual alarm sound
pygame.mixer.music.load(pilarm)

def get_eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_emotion(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48)) / 255.0
    face_array = np.expand_dims(face_resized, axis=0)
    face_array = np.expand_dims(face_array, axis=-1)
    predictions = emotion_model.predict(face_array)
    return emotion_labels[np.argmax(predictions)]

def main():
    cap = cv2.VideoCapture(0)
    eye_closed_time = None
    alarm_triggered = False
    EYE_AR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
    EYE_CLOSED_LIMIT = 120  # Time in seconds (2 min)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = detect_emotion(face)
            cv2.putText(frame, f'Emotion: {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            rects = detector(gray)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                
                left_ear = get_eye_aspect_ratio(left_eye)
                right_ear = get_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < EYE_AR_THRESHOLD:
                    if eye_closed_time is None:
                        eye_closed_time = time.time()
                    else:
                        elapsed_time = time.time() - eye_closed_time
                        if elapsed_time > EYE_CLOSED_LIMIT and not alarm_triggered:
                            pygame.mixer.music.play()
                            alarm_triggered = True
                else:
                    eye_closed_time = None
                    alarm_triggered = False
                    pygame.mixer.music.stop()
                
                cv2.putText(frame, "Eyes Closed" if avg_ear < EYE_AR_THRESHOLD else "Eyes Open", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Face & Emotion Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
