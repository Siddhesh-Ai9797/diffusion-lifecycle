import torch
import torch.nn.functional as F
import time
import os

from src.unet import SimpleUNet
from src.classifier import SimpleClassifier
from src.noise_schedule import T, alphas_cumprod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 16  # smaller batch for faster comparison

alphas_cumprod = alphas_cumprod.to(DEVICE)


def generate_samples(model, dtype):
    """Run reverse process, return generated images + time taken."""
    x = torch.randn(NUM_IMAGES, 3, 32, 32, device=DEVICE, dtype=dtype)

    start = time.time()
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((NUM_IMAGES,), t, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t_batch)

            sqrt_alpha_t = torch.sqrt(alphas_cumprod[t]).to(dtype)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[t]).to(dtype)

            x0_approx = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

            if t > 0:
                sqrt_alpha_prev = torch.sqrt(alphas_cumprod[t - 1]).to(dtype)
                sqrt_one_minus_alpha_prev = torch.sqrt(1 - alphas_cumprod[t - 1]).to(dtype)
                x = sqrt_alpha_prev * x0_approx + sqrt_one_minus_alpha_prev * predicted_noise
            else:
                x = x0_approx
    elapsed = time.time() - start
    return x, elapsed


def get_model_size(model):
    """Save model to disk, check file size in MB."""
    torch.save(model.state_dict(), "outputs/_temp_model.pt")
    size_mb = os.path.getsize("outputs/_temp_model.pt") / (1024 * 1024)
    os.remove("outputs/_temp_model.pt")
    return size_mb


def score_images(images, classifier):
    """Get mean classifier confidence."""
    with torch.no_grad():
        logits = classifier(images.float())  # classifier stays FP32
        probs = F.softmax(logits, dim=1)
        confidences = probs.max(dim=1).values
    return confidences.mean().item()


if __name__ == "__main__":
    classifier = SimpleClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load("outputs/classifier.pt", map_location=DEVICE))
    classifier.eval()

    results = {}

    # ---- FP32 (baseline) ----
    model_fp32 = SimpleUNet().to(DEVICE)
    model_fp32.load_state_dict(torch.load("outputs/model_dpo.pt", map_location=DEVICE))
    model_fp32.eval()

    size_fp32 = get_model_size(model_fp32)
    images_fp32, time_fp32 = generate_samples(model_fp32, torch.float32)
    conf_fp32 = score_images(images_fp32, classifier)

    results["FP32"] = {"size_mb": size_fp32, "time_sec": time_fp32, "confidence": conf_fp32}

    # ---- FP16 ----
    model_fp16 = SimpleUNet().to(DEVICE).half()  # convert weights to FP16
    model_fp16.load_state_dict(torch.load("outputs/model_dpo.pt", map_location=DEVICE))
    model_fp16 = model_fp16.half()
    model_fp16.eval()

    size_fp16 = get_model_size(model_fp16)
    images_fp16, time_fp16 = generate_samples(model_fp16, torch.float16)
    conf_fp16 = score_images(images_fp16, classifier)

    results["FP16"] = {"size_mb": size_fp16, "time_sec": time_fp16, "confidence": conf_fp16}

    # ---- Print comparison table ----
    print(f"{'Precision':<10} {'Size (MB)':<12} {'Time (sec)':<12} {'Confidence':<12}")
    for name, r in results.items():
        print(f"{name:<10} {r['size_mb']:<12.3f} {r['time_sec']:<12.3f} {r['confidence']:<12.3f}")

    torch.save(model_fp16.state_dict(), "outputs/model_fp16.pt")
    print("Saved outputs/model_fp16.pt")