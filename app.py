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

# (Rest of your functions remain the same: detect_landmarks, calculate_eyes_score, calculate_nose_score, predict)

# Gradio interface with disclaimer, description, and Google Analytics
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(),
    title="Face Beauty Rating with Symmetry and Feature Scores",
    description="Upload an image to get a combined beauty score based on deep learning and facial feature scores (eyes, nose). "
                "This model was trained on the SCUT-FBP5500 dataset and uses Dlib for landmark detection.\n\n"
                "Disclaimer: This model is for educational purposes only and should not be taken as a definitive judgment of physical appearance.\n\n"
                "**Note:** Please upload a clear, front-facing photo where the face is fully visible and not obstructed. Ensure good lighting and that the face is not too angled or cropped. The model requires the face to be properly aligned to detect landmarks such as the eyes and nose accurately. Failure to do so may result in errors or inaccurate scores.",
    allow_flagging="never",
    live=False,
    analytics_enabled=True,
    css="",  # Remove any previous CSS
    js="""
    // Google tag (gtag.js)
    var script = document.createElement('script');
    script.src = 'https://www.googletagmanager.com/gtag/js?id=G-RZF99L2LWQ';
    script.async = true;
    document.head.appendChild(script);

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
    """
)

# Launch the app
iface.launch()