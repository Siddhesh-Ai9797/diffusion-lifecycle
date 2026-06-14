import torch
import torch.nn.functional as F
import tensorrt as trt
import numpy as np
import time
import os
import io

from src.unet import SimpleUNet
from src.noise_schedule import T, alphas_cumprod
from src.classifier import SimpleClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 16
NUM_WARMUP = 3
NUM_TIMING = 5
ONNX_PATH = "outputs/unet.onnx"
TRT_PATH = "outputs/unet.trt"

torch.set_float32_matmul_precision('high')
alphas_cumprod = alphas_cumprod.to(DEVICE)

# Load classifier
classifier = SimpleClassifier().to(DEVICE)
classifier.load_state_dict(torch.load("outputs/classifier.pt", map_location=DEVICE))
classifier.eval()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# Step 1: Export to ONNX
# ============================================================
def export_to_onnx(model, path):
    print(f"\nExporting to ONNX: {path}")
    model.eval()

    dummy_x = torch.randn(NUM_IMAGES, 3, 32, 32, device=DEVICE)
    dummy_t = torch.randint(0, T, (NUM_IMAGES,), device=DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_x, dummy_t),
            path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["x", "t"],
            output_names=["noise_pred"],
            dynamic_axes={
                "x": {0: "batch_size"},
                "t": {0: "batch_size"},
                "noise_pred": {0: "batch_size"}
            },
            dynamo=False  # force old exporter — embeds weights in single file
        )

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ONNX exported: {size_mb:.2f} MB")
    return path


# ============================================================
# Step 2: Build TensorRT engine from ONNX
# ============================================================
def build_trt_engine(onnx_path, trt_path, fp16=True):
    print(f"\nBuilding TensorRT engine (fp16={fp16})...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  TRT parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled")

    # Add optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape("x",
    min=(1, 3, 32, 32),
    opt=(NUM_IMAGES, 3, 32, 32),
    max=(NUM_IMAGES, 3, 32, 32)
)
    profile.set_shape("t",
    min=(1,),
    opt=(NUM_IMAGES,),
    max=(NUM_IMAGES,)
)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(trt_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(trt_path) / (1024 * 1024)
    print(f"  TRT engine saved: {size_mb:.2f} MB")
    return trt_path


# ============================================================
# Step 3: TensorRT inference wrapper
# ============================================================
class TRTModel:
    def __init__(self, trt_path):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(trt_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        print(f"  TRT engine loaded: {trt_path}")

    def infer(self, x, t):
        """Run one forward pass through TensorRT engine."""
        x_np = x.cpu().numpy().astype(np.float32)
        t_np = t.cpu().numpy().astype(np.int64)

        x_tensor = torch.from_numpy(x_np).to(DEVICE)
        t_tensor = torch.from_numpy(t_np).to(DEVICE)

        output = torch.zeros(x.shape, device=DEVICE, dtype=torch.float32)

        self.context.set_tensor_address("x", x_tensor.data_ptr())
        self.context.set_tensor_address("t", t_tensor.data_ptr())
        self.context.set_tensor_address("noise_pred", output.data_ptr())

        self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()
        return output


# ============================================================
# Step 4: Sampling loops
# ============================================================
def run_sampling_pytorch(model, num_images=NUM_IMAGES):
    """Standard PyTorch sampling."""
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


def run_sampling_trt(trt_model, num_images=NUM_IMAGES):
    """TensorRT sampling."""
    x = torch.randn(num_images, 3, 32, 32, device=DEVICE)
    for t in reversed(range(T)):
        t_batch = torch.full((num_images,), t, device=DEVICE, dtype=torch.long)
        predicted_noise = trt_model.infer(x, t_batch)
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


def benchmark(label, sample_fn):
    """Warmup then time, return results."""
    print(f"\n  Warming up {label}...")
    for _ in range(NUM_WARMUP):
        images = sample_fn()
    torch.cuda.synchronize()

    print(f"  Timing {label}...")
    times = []
    for i in range(NUM_TIMING):
        torch.cuda.synchronize()
        start = time.time()
        images = sample_fn()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)

    with torch.no_grad():
        logits = classifier(images.float())
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values.mean().item()

    return {"avg_time_sec": round(avg_time, 3), "confidence": round(confidence, 4)}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    results = {}

    # Load PyTorch model
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load("outputs/model_epoch99.pt", map_location=DEVICE))
    model.eval()

    # ---- 1. PyTorch Eager ----
    print("=" * 50)
    print("Mode 1: PyTorch Eager")
    results["pytorch_eager"] = benchmark("pytorch_eager", lambda: run_sampling_pytorch(model))

    # ---- 2. torch.compile reduce-overhead (our best from Option B) ----
    print("=" * 50)
    print("Mode 2: torch.compile (reduce-overhead)")
    model_compiled = torch.compile(model, mode="reduce-overhead")
    results["torch_compile"] = benchmark("torch_compile", lambda: run_sampling_pytorch(model_compiled))

    # ---- 3. TensorRT FP32 ----
    print("=" * 50)
    print("Mode 3: TensorRT FP32")
    export_to_onnx(model, ONNX_PATH)
    build_trt_engine(ONNX_PATH, "outputs/unet_fp32.trt", fp16=False)
    trt_model_fp32 = TRTModel("outputs/unet_fp32.trt")
    results["tensorrt_fp32"] = benchmark("tensorrt_fp32", lambda: run_sampling_trt(trt_model_fp32))

    # ---- 4. TensorRT FP16 ----
    print("=" * 50)
    print("Mode 4: TensorRT FP16")
    build_trt_engine(ONNX_PATH, "outputs/unet_fp16.trt", fp16=True)
    trt_model_fp16 = TRTModel("outputs/unet_fp16.trt")
    results["tensorrt_fp16"] = benchmark("tensorrt_fp16", lambda: run_sampling_trt(trt_model_fp16))

    # ---- Print table ----
    print(f"\n{'='*60}")
    print(f"{'Mode':<25} {'Time(s)':>10} {'Conf':>8} {'Speedup':>10}")
    print(f"{'='*60}")
    baseline_time = results["pytorch_eager"]["avg_time_sec"]
    for name, r in results.items():
        speedup = baseline_time / r["avg_time_sec"]
        print(f"{name:<25} {r['avg_time_sec']:>10.3f} {r['confidence']:>8.4f} {speedup:>9.2f}x")
    print(f"{'='*60}")

    torch.save(results, "outputs/tensorrt_benchmark_results.pt")
    print("\nSaved outputs/tensorrt_benchmark_results.pt")