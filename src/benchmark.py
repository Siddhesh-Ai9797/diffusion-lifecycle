import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import mlflow
import os

from src.unet import SimpleUNet
from src.unet_attention import SimpleUNetAttention
from src.unet_linear_attention import SimpleUNetLinearAttention
from src.unet_mamba import SimpleUNetMamba
from src.dataset import get_dataloader
from src.noise_schedule import T, alphas_cumprod, add_noise
from src.classifier import SimpleClassifier
from torchvision.utils import save_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 64
LR = 2e-4
NUM_SAMPLE = 16  # images to generate for quality check

alphas_cumprod = alphas_cumprod.to(DEVICE)

# ---- Classifier (our quality judge) ----
classifier = SimpleClassifier().to(DEVICE)
classifier.load_state_dict(torch.load("outputs/classifier.pt", map_location=DEVICE))
classifier.eval()


def sample_images(model, num_images=NUM_SAMPLE):
    """Generate images using reverse process, return classifier confidence."""
    model.eval()
    x = torch.randn(num_images, 3, 32, 32, device=DEVICE)
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((num_images,), t, device=DEVICE, dtype=torch.long)
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

    with torch.no_grad():
        logits = classifier(x.float())
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values.mean().item()

    return x, confidence


def get_gpu_memory():
    """Return current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def train_and_benchmark(name, model, dataloader):
    """Train model for EPOCHS, return benchmark results."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    torch.cuda.reset_peak_memory_stats()  # reset memory counter

    epoch_times = []
    final_loss = None

    mlflow.set_experiment("architecture-benchmark")
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("params", sum(p.numel() for p in model.parameters()))

        step = 0
        for epoch in range(EPOCHS):
            model.train()
            epoch_start = time.time()

            for images, _ in dataloader:
                images = images.to(DEVICE)
                batch_size = images.shape[0]

                t = torch.randint(0, T, (batch_size,), device=DEVICE)
                noisy_images, true_noise = add_noise(images, t, alphas_cumprod)
                predicted_noise = model(noisy_images, t)
                loss = loss_fn(predicted_noise, true_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                final_loss = loss.item()
                step += 1

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch} | Loss: {final_loss:.4f} | Time: {epoch_time:.1f}s")
                mlflow.log_metric("loss", final_loss, step=epoch)

        # Generate samples + score quality
        print(f"  Generating samples for quality check...")
        samples, confidence = sample_images(model)
        gpu_mem = get_gpu_memory()

        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        results = {
            "params": sum(p.numel() for p in model.parameters()),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
            "final_loss": round(final_loss, 4),
            "confidence": round(confidence, 4),
            "gpu_memory_mb": round(gpu_mem, 1),
        }

        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("gpu_memory_mb", gpu_mem)
        mlflow.log_metric("avg_epoch_time", avg_epoch_time)

        # Save sample grid
        os.makedirs("outputs/benchmark", exist_ok=True)
        imgs = (samples.clamp(-1, 1) + 1) / 2
        save_image(imgs, f"outputs/benchmark/{name}_samples.png", nrow=4)
        print(f"  Saved outputs/benchmark/{name}_samples.png")

    return results


if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=BATCH_SIZE)

    models = {
        "baseline_cnn":       SimpleUNet(),
        "standard_attention": SimpleUNetAttention(),
        "linear_attention":   SimpleUNetLinearAttention(),
        "mamba_ssm":          SimpleUNetMamba(),
    }

    all_results = {}
    for name, model in models.items():
        results = train_and_benchmark(name, model, dataloader)
        all_results[name] = results

    # Print final comparison table
    print(f"\n{'='*75}")
    print(f"{'Model':<25} {'Params':>10} {'Epoch(s)':>10} {'Loss':>8} {'Conf':>8} {'GPU(MB)':>10}")
    print(f"{'='*75}")
    for name, r in all_results.items():
        print(f"{name:<25} {r['params']:>10,} {r['avg_epoch_time_sec']:>10.2f} "
              f"{r['final_loss']:>8.4f} {r['confidence']:>8.4f} {r['gpu_memory_mb']:>10.1f}")
    print(f"{'='*75}")

    # Save results
    torch.save(all_results, "outputs/benchmark_results.pt")
    print("\nSaved outputs/benchmark_results.pt")