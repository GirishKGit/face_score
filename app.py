import gradio as gr
from fastai.vision.all import *
import torch

# Load the trained model (make sure the model file is in the same directory or update the path)
learn = load_learner('beauty_model_finetuned_export.pkl')

# Image preprocessing and prediction function
def predict(image):
    # Convert to a Fastai PILImage for prediction
    img = PILImage.create(image)
    
    # Get prediction from the model (the prediction score is in the second position)
    pred_score = learn.predict(img)[1].item()  # Extract the score (as float)

    # Return the beauty score with the range clarified
    return f"Predicted beauty score: {pred_score:.2f} / 5"

# Create Gradio interface with updated components
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Updated from gr.inputs.Image to gr.Image
    outputs=gr.Text(),  # Updated from gr.outputs.Text to gr.Text
    title="Face Beauty Rating",
    description="Upload an image to get a beauty score prediction (out of 5) from a fine-tuned ResNet50 model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
