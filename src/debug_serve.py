import torch
from src.unet import SimpleUNet
from src.noise_schedule import T, alphas_cumprod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load("outputs/model_dpo.pt", map_location=DEVICE))
model.eval()

alphas_cumprod = alphas_cumprod.to(DEVICE)

num_images = 4
x = torch.randn(num_images, 3, 32, 32, device=DEVICE, dtype=torch.float32)

with torch.no_grad():
    for t in reversed(range(T)):
        t_batch = torch.full((num_images,), t, device=DEVICE, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        sqrt_alpha_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[t])

        x0_approx = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

        if t > 0:
            sqrt_alpha_prev = torch.sqrt(alphas_cumprod[t - 1]).half()
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alphas_cumprod[t - 1])
            x = sqrt_alpha_prev * x0_approx + sqrt_one_minus_alpha_prev * predicted_noise
        else:
            x = x0_approx

        # Print every 200 steps to see progression
        if t % 200 == 0:
            print(f"t={t}: x min={x.min().item():.4f}, max={x.max().item():.4f}, "
                  f"mean={x.mean().item():.4f}, has_nan={torch.isnan(x).any().item()}")
            
    from torchvision.utils import save_image
    images = (x.clamp(-1, 1) + 1) / 2
    images = images.float()
    print("After normalize - min:", images.min().item(), "max:", images.max().item())
    save_image(images, "outputs/debug_test1.png", nrow=2)
    print("Saved outputs/debug_test1.png")

print("\nFinal x stats:")
print("min:", x.min().item(), "max:", x.max().item(), "mean:", x.mean().item())
print("Sample pixel values (first image, first channel, 3x3 corner):")
print(x[0, 0, :3, :3])