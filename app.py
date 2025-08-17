import gradio as gr
from diffusers import DiffusionPipeline
import torch
import os

# Configuration 
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_MODEL_ID = "roshanVarghese/TextToImageShoe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Pipeline 
print("Loading base model...")
pipe = DiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16
).to(DEVICE)

print(f"Loading LoRA weights from {LORA_MODEL_ID}...")
try:
    pipe.load_lora_weights(LORA_MODEL_ID)
    pipe.fuse_lora() 
    print("LoRA weights loaded and fused successfully.")
except Exception as e:
    print(f"Could not load LoRA weights. Running with base model only. Error: {e}")

pipe.unet.eval()

# Define the Generation Function 
def generate(prompt, guidance_scale=7.5, num_steps=50):
    with torch.no_grad():
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps)
        ).images[0]
    return image

# Create the Gradio Interface 
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="a photo of a high-top sneaker, futuristic design"),
        gr.Slider(minimum=1, maximum=20, step=0.5, value=7.5, label="Guidance Scale"),
        gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Inference Steps")
    ],
    outputs=gr.Image(type="pil"),
    title="Generative AI Shoe Generator",
    description="Enter a prompt to generate a unique shoe design using a Stable Diffusion model fine-tuned with LoRA on the Zappos dataset.",
    allow_flagging="never",
    examples=[
        ["a photo of a running shoe, vibrant colors"],
        ["a photo of a leather boot, classic style"],
        ["a photo of a sandal, minimalist design"],
    ]
)

# Launching the App 
demo.launch()
