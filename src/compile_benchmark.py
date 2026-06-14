import torch
import time
import torch.nn.functional as F
import os

from src.unet import SimpleUNet
from src.noise_schedule import T, alphas_cumprod
from src.classifier import SimpleClassifier

torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 16
NUM_WARMUP = 3    # warmup runs before timing (compiler needs a few runs to optimize)
NUM_TIMING = 5    # timing runs to average

alphas_cumprod = alphas_cumprod.to(DEVICE)

# Load classifier for quality check
classifier = SimpleClassifier().to(DEVICE)
classifier.load_state_dict(torch.load("outputs/classifier.pt", map_location=DEVICE))
classifier.eval()


def run_sampling(model, num_images=NUM_IMAGES):
    """One full reverse diffusion pass."""
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
    return x


def benchmark_model(model, label):
    """Warmup then time NUM_TIMING runs, return avg time + confidence."""
    print(f"\n  Warming up {label}...")
    for _ in range(NUM_WARMUP):
        _ = run_sampling(model)
    torch.cuda.synchronize()

    print(f"  Timing {label}...")
    times = []
    for i in range(NUM_TIMING):
        torch.cuda.synchronize()
        start = time.time()
        images = run_sampling(model)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)

    # Quality check
    with torch.no_grad():
        logits = classifier(images.float())
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values.mean().item()

    torch.cuda.reset_peak_memory_stats()
    _ = run_sampling(model)
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "avg_time_sec": round(avg_time, 3),
        "confidence": round(confidence, 4),
        "gpu_memory_mb": round(gpu_mem, 1)
    }


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    results = {}

    # ---- 1. Eager mode (baseline) ----
    print("=" * 50)
    print("Mode 1: Eager (default PyTorch)")
    model_eager = SimpleUNet().to(DEVICE)
    model_eager.load_state_dict(
        torch.load("outputs/model_epoch99.pt", map_location=DEVICE)
    )
    model_eager.eval()
    results["eager"] = benchmark_model(model_eager, "eager")

    # ---- 2. torch.compile default ----
    print("=" * 50)
    print("Mode 2: torch.compile (default)")
    model_compiled = SimpleUNet().to(DEVICE)
    model_compiled.load_state_dict(
        torch.load("outputs/model_epoch99.pt", map_location=DEVICE)
    )
    model_compiled.eval()
    model_compiled = torch.compile(model_compiled)
    results["compiled_default"] = benchmark_model(model_compiled, "compiled_default")

    # ---- 3. torch.compile reduce-overhead ----
    print("=" * 50)
    print("Mode 3: torch.compile (reduce-overhead)")
    model_reduced = SimpleUNet().to(DEVICE)
    model_reduced.load_state_dict(
        torch.load("outputs/model_epoch99.pt", map_location=DEVICE)
    )
    model_reduced.eval()
    model_reduced = torch.compile(model_reduced, mode="reduce-overhead")
    results["compiled_reduce_overhead"] = benchmark_model(model_reduced, "compiled_reduce_overhead")

    # ---- 4. torch.compile max-autotune ----
    print("=" * 50)
    print("Mode 4: torch.compile (max-autotune)")
    model_maxauto = SimpleUNet().to(DEVICE)
    model_maxauto.load_state_dict(
        torch.load("outputs/model_epoch99.pt", map_location=DEVICE)
    )
    model_maxauto.eval()
    model_maxauto = torch.compile(model_maxauto, mode="max-autotune")
    results["compiled_max_autotune"] = benchmark_model(model_maxauto, "compiled_max_autotune")

    # ---- Print table ----
    print(f"\n{'='*65}")
    print(f"{'Mode':<30} {'Time(s)':>10} {'Conf':>8} {'GPU(MB)':>10}")
    print(f"{'='*65}")
    for name, r in results.items():
        speedup = results["eager"]["avg_time_sec"] / r["avg_time_sec"]
        print(f"{name:<30} {r['avg_time_sec']:>10.3f} {r['confidence']:>8.4f} "
              f"{r['gpu_memory_mb']:>10.1f}  ({speedup:.2f}x)")
    print(f"{'='*65}")

    torch.save(results, "outputs/compile_benchmark_results.pt")
    print("\nSaved outputs/compile_benchmark_results.pt")