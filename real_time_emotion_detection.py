"""
Real-time emotion detection using webcam
"""
import warnings
import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
import keras

# Configuration
IMG_SIZE = 48
MODEL_PATH = 'emotion_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

# Color mapping for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Green
    'Sad': (255, 0, 0),        # Blue
    'Suprise': (0, 165, 255)   # Orange
}

class EmotionDetector:
    """Emotion detector class for real-time face emotion recognition."""
    
    def __init__(self, model_path=MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
        """
        Initialize the emotion detector.
        
        Args:
            model_path: Path to trained model file
            label_encoder_path: Path to label encoder pickle file
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
        
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load label encoder
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            print("Warning: Could not load face cascade. Trying alternative path...")
            # Try alternative path
            alt_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
            if os.path.exists(alt_path):
                self.face_cascade = cv2.CascadeClassifier(alt_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Could not load face detection cascade. Please check OpenCV installation.")
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input.
        
        Args:
            face_img: Face image (BGR or grayscale)
        
        Returns:
            Preprocessed image tensor ready for model prediction
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        
        # Reshape for model input
        face_img = face_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return face_img
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from face image.
        
        Args:
            face_img: Face image (BGR or grayscale)
        
        Returns:
            emotion: Predicted emotion label
            confidence: Prediction confidence (0-1)
            all_predictions: Array of all class probabilities
        """
        # Preprocess
        processed = self.preprocess_face(face_img)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        
        # Get predicted class and confidence
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        # Decode emotion label
        emotion = self.label_encoder.inverse_transform([emotion_idx])[0]
        
        return emotion, confidence, predictions[0]
    
    def detect_faces(self, frame):
        """
        Detect faces in frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence, all_predictions):
        """
        Draw emotion information on frame.
        
        Args:
            frame: Frame to draw on
            x, y, w, h: Face bounding box coordinates
            emotion: Predicted emotion label
            confidence: Prediction confidence
            all_predictions: All class probabilities (unused but kept for API consistency)
        """
        # Draw rectangle around face
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        emotion_text = f"{emotion}: {confidence:.2%}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            emotion_text, font, font_scale, thickness
        )
        
        # Draw background for text
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 10),
            (x + text_width, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            emotion_text,
            (x, y - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Draw confidence bar
        bar_width = int(w * confidence)
        cv2.rectangle(
            frame,
            (x, y + h + 5),
            (x + bar_width, y + h + 15),
            color,
            -1
        )
        cv2.rectangle(
            frame,
            (x, y + h + 5),
            (x + w, y + h + 15),
            (255, 255, 255),
            1
        )

def main():
    """
    Main function for real-time emotion detection.
    
    Initializes detector, opens webcam, and runs real-time emotion detection loop.
    """
    print("Initializing Emotion Detector...")
    try:
        detector = EmotionDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nReal-time Emotion Detection Started!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence, all_predictions = detector.predict_emotion(face_roi)
            
            # Draw information on frame
            detector.draw_emotion_info(frame, x, y, w, h, emotion, confidence, all_predictions)
        
        # Add FPS counter
        frame_count += 1
        fps_text = f"FPS: {frame_count}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Display frame
        cv2.imshow('Real-time Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"captured_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nEmotion detection stopped.")

if __name__ == "__main__":
    main()

