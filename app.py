import gradio as gr
from diffusers import DiffusionPipeline
import torch
import os

# --- Configuration ---
MODEL_ID = "roshanVarghese/TextToImageShoe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Load the Pipeline ---
print(f"Loading fine-tuned model from {MODEL_ID}...")

# The 'custom_pipeline' argument forces the use of the standard pipeline code,
# which is a robust way to bypass configuration errors.
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    custom_pipeline="stable-diffusion", # This is the key fix
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False
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
        gr.Textbox(label="Prompt", value="a futuristic running shoe, neon accents"),
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
