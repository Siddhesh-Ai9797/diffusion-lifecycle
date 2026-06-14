import torch
from torchvision.utils import save_image
import os

from src.unet import SimpleUNet
from src.noise_schedule import T, alphas_cumprod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs/model_epoch99.pt"  # change to your latest checkpoint
NUM_IMAGES = 16

# Load model
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

alphas_cumprod = alphas_cumprod.to(DEVICE)

# Start from pure noise
x = torch.randn(NUM_IMAGES, 3, 32, 32, device=DEVICE)

with torch.no_grad():
    for t in reversed(range(T)):
        t_batch = torch.full((NUM_IMAGES,), t, device=DEVICE, dtype=torch.long)

        predicted_noise = model(x, t_batch)

        sqrt_alpha_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[t])

        x0_approx = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

        if t > 0:
            sqrt_alpha_prev = torch.sqrt(alphas_cumprod[t - 1])
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alphas_cumprod[t - 1])
            x = sqrt_alpha_prev * x0_approx + sqrt_one_minus_alpha_prev * predicted_noise
        else:
            x = x0_approx

# Rescale from [-1, 1] back to [0, 1] for saving as image
x = (x.clamp(-1, 1) + 1) / 2

os.makedirs("outputs", exist_ok=True)
save_image(x, "outputs/samplesnew.png", nrow=4)
print("Saved outputs/samplesnew.png")