import gradio as gr
from fastai.vision.all import *
import torch

# Load the trained model (make sure the model file is in the same directory or update the path)
learn = load_learner('beauty_model_finetuned_export.pkl')

# Image preprocessing function
def predict(image):
    # Convert to a Fastai PILImage for prediction
    img = PILImage.create(image)
    
    # Get prediction from the model
    pred_score = learn.predict(img)
    
    # Return the beauty score
    return f"Predicted beauty score: {pred_score[0]:.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="pil"),
    outputs="text",
    title="Face Beauty Rating",
    description="Upload an image to get a beauty score prediction from a fine-tuned ResNet50 model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
