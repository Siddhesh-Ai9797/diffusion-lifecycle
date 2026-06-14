import torch
import io
import time
import json
import os
import torch.nn.functional as F
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision.utils import save_image

from src.unet import SimpleUNet
from src.classifier import SimpleClassifier
from src.noise_schedule import T, alphas_cumprod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS_FILE = "outputs/metrics.json"

app = FastAPI(title="Diffusion Model API")

# ---- Load models once at startup ----
model = SimpleUNet().to(DEVICE).half()
model.load_state_dict(torch.load("outputs/model_fp16.pt", map_location=DEVICE))
model.eval()

classifier = SimpleClassifier().to(DEVICE)
classifier.load_state_dict(torch.load("outputs/classifier.pt", map_location=DEVICE))
classifier.eval()

alphas_cumprod = alphas_cumprod.to(DEVICE)

# ---- In-memory metrics store ----
request_log = []


def log_request(num_images, latency_ms, gpu_memory_mb, confidence):
    """Log one request's metrics to memory + disk."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_images": num_images,
        "latency_ms": round(latency_ms, 2),
        "gpu_memory_mb": round(gpu_memory_mb, 2),
        "confidence": round(confidence, 4),
    }
    request_log.append(entry)

    # Append to JSON file
    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(request_log, f, indent=2)


def compute_percentile(values, p):
    """Compute p-th percentile of a list of values."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


# ---- Endpoints ----

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "total_requests": len(request_log)
    }


@app.post("/generate")
def generate(num_images: int = 1):
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    # Reverse diffusion
    x = torch.randn(num_images, 3, 32, 32, device=DEVICE, dtype=torch.float16)
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((num_images,), t, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            sqrt_alpha_t = torch.sqrt(alphas_cumprod[t]).half()
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[t]).half()
            x0_approx = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            if t > 0:
                sqrt_alpha_prev = torch.sqrt(alphas_cumprod[t - 1]).half()
                sqrt_one_minus_alpha_prev = torch.sqrt(1 - alphas_cumprod[t - 1]).half()
                x = sqrt_alpha_prev * x0_approx + sqrt_one_minus_alpha_prev * predicted_noise
            else:
                x = x0_approx

    # Measure latency + GPU memory
    latency_ms = (time.time() - start_time) * 1000
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Score quality
    with torch.no_grad():
        logits = classifier(x.float())
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values.mean().item()

    # Log metrics
    log_request(num_images, latency_ms, gpu_memory_mb, confidence)

    # Return image
    images = (x.clamp(-1, 1) + 1) / 2
    images = images.float()
    buf = io.BytesIO()
    save_image(images, buf, format="PNG", nrow=max(1, int(num_images**0.5)))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Latency-Ms": str(round(latency_ms, 2)),
            "X-GPU-Memory-MB": str(round(gpu_memory_mb, 2)),
            "X-Confidence": str(round(confidence, 4)),
        }
    )


@app.get("/metrics")
def metrics():
    if not request_log:
        return JSONResponse({"message": "No requests yet"})

    latencies = [r["latency_ms"] for r in request_log]
    confidences = [r["confidence"] for r in request_log]
    gpu_memories = [r["gpu_memory_mb"] for r in request_log]

    return {
        "total_requests": len(request_log),
        "latency_ms": {
            "p50": compute_percentile(latencies, 50),
            "p95": compute_percentile(latencies, 95),
            "p99": compute_percentile(latencies, 99),
            "mean": round(sum(latencies) / len(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "confidence": {
            "mean": round(sum(confidences) / len(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
        },
        "gpu_memory_mb": {
            "mean": round(sum(gpu_memories) / len(gpu_memories), 2),
            "max": round(max(gpu_memories), 2),
        },
        "recent_requests": request_log[-5:]
    }


@app.delete("/metrics/reset")
def reset_metrics():
    request_log.clear()
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
    return {"message": "Metrics reset"}