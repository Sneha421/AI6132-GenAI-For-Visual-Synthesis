import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
from torchvision import transforms
from pathlib import Path

# -------------------- Step 1: Define paths -----------------------------
checkpoint_url = "https://huggingface.co/snehabalaji74/ControlNet/resolve/main/LastConstrolNetcheckpoint.pt"

condition_paths = [
    "bare.jpg",
    # add paths to 5 more condition images
]

output_path = Path("sd_controlnet_output.png")

# -------------------- Step 2: Load condition images --------------------
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

conditions = []
for p in condition_paths:
    img = Image.open(p).convert("RGB")
    conditions.append(transform(img))
conditions = torch.stack(conditions).unsqueeze(0)  # [1, 6, C, H, W]

# -------------------- Step 3: Load ControlNet --------------------------
# Use a template architecture
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Load your checkpoint weights (strict=False to avoid mismatches)
state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
controlnet.load_state_dict(state_dict['model'], strict=False)
controlnet = controlnet

# -------------------- Step 4: Load Stable Diffusion v1-5 ----------------
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# -------------------- Step 5: Generate image --------------------------
prompt = "Apply natural foundation to the face in the image"

result = pipe( prompt=prompt,image=[Image.open(p).convert("RGB") for p in condition_paths]
    # prompt=prompt,
    # control_image=[Image.open(p).convert("RGB") for p in condition_paths],  # list of PIL images
    # controlnet_conditioning_scale=1.0,
    # num_inference_steps=50,
    # guidance_scale=7.5
).images[0]

result.save(output_path)
print(f"Saved final image to {output_path}")
