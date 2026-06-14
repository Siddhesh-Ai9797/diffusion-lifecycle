import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

from src.unet import SimpleUNet
from src.classifier import SimpleClassifier
from src.noise_schedule import T, alphas_cumprod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 100

# Load models
diffusion_model = SimpleUNet().to(DEVICE)
diffusion_model.load_state_dict(torch.load("outputs/model_epoch99.pt", map_location=DEVICE))
diffusion_model.eval()

classifier = SimpleClassifier().to(DEVICE)
classifier.load_state_dict(torch.load("outputs/classifier.pt", map_location=DEVICE))
classifier.eval()

alphas_cumprod = alphas_cumprod.to(DEVICE)

# Generate images (same sampling loop as src/sample.py)
x = torch.randn(NUM_IMAGES, 3, 32, 32, device=DEVICE)

with torch.no_grad():
    for t in reversed(range(T)):
        t_batch = torch.full((NUM_IMAGES,), t, device=DEVICE, dtype=torch.long)
        predicted_noise = diffusion_model(x, t_batch)

        sqrt_alpha_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[t])

        x0_approx = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

        if t > 0:
            sqrt_alpha_prev = torch.sqrt(alphas_cumprod[t - 1])
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alphas_cumprod[t - 1])
            x = sqrt_alpha_prev * x0_approx + sqrt_one_minus_alpha_prev * predicted_noise
        else:
            x = x0_approx

    # x is now our generated images, range [-1, 1]

    # Score with classifier
    logits = classifier(x)
    probs = F.softmax(logits, dim=1)
    confidences = probs.max(dim=1).values  # shape: (NUM_IMAGES,)

# Sort by confidence
sorted_indices = torch.argsort(confidences, descending=True)

print(f"Confidence range: min={confidences.min().item():.3f}, max={confidences.max().item():.3f}, mean={confidences.mean().item():.3f}")

# Save the generated images + confidences for next step (DPO)
images_save = (x.clamp(-1, 1) + 1) / 2
os.makedirs("outputs", exist_ok=True)
torch.save({"images": x.cpu(), "confidences": confidences.cpu()}, "outputs/scored_samples.pt")
save_image(images_save, "outputs/scored_samples_grid.png", nrow=10)
print("Saved outputs/scored_samples.pt and outputs/scored_samples_grid.png")