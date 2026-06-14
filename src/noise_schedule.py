import torch

# --- Noise schedule setup ---
T = 1000

betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def add_noise(image, t, alphas_cumprod):
    noise = torch.randn_like(image)
    sqrt_alpha = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    noisy_image = sqrt_alpha * image + sqrt_one_minus_alpha * noise
    return noisy_image, noise


if __name__ == "__main__":
    print(betas.shape, alphas.shape, alphas_cumprod.shape)
    print(alphas_cumprod[0], alphas_cumprod[-1])