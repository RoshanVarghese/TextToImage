import gradio as gr
from diffusers import DiffusionPipeline
import torch
import os
# We don't need the safety_checker import when loading a full pipeline this way

# --- Configuration ---
MODEL_ID = "roshanVarghese/TextToImageShoe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Load the Pipeline ---
print(f"Loading full fine-tuned model from {MODEL_ID}...")

# Load the pipeline and disable the safety checker to fix the error
# Setting safety_checker to None tells the pipeline to not load that component
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
).to(DEVICE)

print("Model loaded successfully.")
pipe.unet.eval()

# --- Define the Generation Function ---
def generate(prompt, guidance_scale=7.5, num_steps=50):
    with torch.no_grad():
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps)
        ).images[0]
    return image

# --- Create the Gradio Interface ---
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="a photo of a high-top sneaker, futuristic design"),
        gr.Slider(minimum=1, maximum=20, step=0.5, value=7.5, label="Guidance Scale"),
        gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Inference Steps")
    ],
    outputs=gr.Image(type="pil"),
    title="Generative AI Shoe Generator",
    description="Enter a prompt to generate a unique shoe design using a fully fine-tuned Stable Diffusion model.",
    allow_flagging="never",
    examples=[
        ["a photo of a running shoe, vibrant colors"],
        ["a photo of a leather boot, classic style"],
    ]
)

# --- Launch the App ---
demo.launch()
