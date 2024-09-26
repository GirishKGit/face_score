import gradio as gr
from fastai.vision.all import *
import dlib
import cv2
import numpy as np

# Load Dlib's pre-trained facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the trained deep learning model
learn = load_learner('beauty_model_finetuned_export.pkl')

# Function to detect landmarks and calculate feature scores (eyes, nose, etc.)
def detect_landmarks(image):
    # Convert to grayscale for Dlib
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # Extract landmark points
    landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    
    # Calculate individual feature scores (eyes, nose)
    eyes_score = calculate_eyes_score(landmarks_points)
    nose_score = calculate_nose_score(landmarks_points)
    
    return eyes_score, nose_score

# Function to calculate symmetry score for eyes
def calculate_eyes_score(landmarks):
    left_eye = landmarks[36:42]  # Left eye landmarks
    right_eye = landmarks[42:48]  # Right eye landmarks
    
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    
    # Calculate distance between left and right eyes (symmetry)
    symmetry_score = np.linalg.norm(left_eye_center - right_eye_center)
    
    # Normalize to a 1-5 scale
    normalized_eyes_score = 5 - (symmetry_score / 10)  # Adjust divisor based on observation
    return max(1, min(5, normalized_eyes_score))

# Function to calculate score for the nose (can use ratio-based measurements)
def calculate_nose_score(landmarks):
    # Example: distance between the nose tip and other facial features (e.g., nose bridge)
    nose_tip = landmarks[30]  # Nose tip
    nose_bridge = np.mean([landmarks[27], landmarks[28], landmarks[29]], axis=0)  # Nose bridge
    
    # Calculate symmetry or proportion
    nose_score = np.linalg.norm(nose_tip - nose_bridge)
    
    # Normalize to a 1-5 scale
    normalized_nose_score = 5 - (nose_score / 10)  # Adjust divisor based on observation
    return max(1, min(5, normalized_nose_score))

# Function to predict combined beauty score (eyes, nose, deep learning score, and total)
def predict(image):
    # Resize the image to a consistent size (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Deep learning model prediction
    img = PILImage.create(image)
    deep_learning_score = learn.predict(img)[1].item()

    # Symmetry score from landmarks (eyes and nose)
    symmetry_scores = detect_landmarks(image)
    
    if symmetry_scores is None:
        return "Error: Could not detect facial landmarks."
    
    eyes_score, nose_score = symmetry_scores
    
    # Calculate facial feature score as an average of eyes and nose
    facial_feature_score = (eyes_score + nose_score) / 2
    
    # Calculate total average score (Simple Average)
    total_score = (deep_learning_score + facial_feature_score) / 2
    
    # Return detailed scores for each feature and total score
    return (f"Deep Learning Score: {deep_learning_score:.2f} / 5\n"
            f"Facial Feature Score: {facial_feature_score:.2f} / 5 (Eyes Score: {eyes_score:.2f} / 5, Nose Score: {nose_score:.2f} / 5)\n"
            f"Total Combined Score: {total_score:.2f} / 5")

# Gradio interface with disclaimer and description
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Upload an image in JPG or PNG format
    outputs=gr.Text(),
    title="Face Beauty Rating with Symmetry and Feature Scores",
    description="Upload an image to get a combined beauty score based on deep learning and facial feature scores (eyes, nose). "
                "This model was trained on the SCUT-FBP5500 dataset and uses Dlib for landmark detection.\n\n"
                "Disclaimer: This model is for educational purposes only and should not be taken as a definitive judgment of physical appearance.\n\n"
                "**Note:** Please upload a clear, front-facing photo where the face is fully visible and not obstructed. Ensure good lighting and that the face is not too angled or cropped. The model requires the face to be properly aligned to detect landmarks such as the eyes and nose accurately. Failure to do so may result in errors or inaccurate scores.",
    allow_flagging="never",
    live=False,  # Add a Submit button
    analytics_enabled=True,
    css="""
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-RZF99L2LWQ"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-RZF99L2LWQ');
      
      // Custom event tracking
      document.addEventListener('DOMContentLoaded', (event) => {
        const predictButton = document.querySelector('button.primary');
        if (predictButton) {
          predictButton.addEventListener('click', () => {
            gtag('event', 'predict_button_click', {
              'event_category': 'User Interaction',
              'event_label': 'Predict Button Clicked'
            });
          });
        }
      });
    </script>
    """
)

# Launch the app
iface.launch()