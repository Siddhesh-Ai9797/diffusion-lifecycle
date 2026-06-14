# Diffusion Model Lifecycle: Train → Align → Quantize → Serve

> End-to-end production ML pipeline implementing DDPM diffusion model training, DPO preference alignment, inference optimization (TensorRT 2.57x speedup), and Dockerized FastAPI serving — with full MLOps stack (MLflow, Docker, Kubernetes, Airflow).

---

## 🎯 Project Overview

This project builds a **complete diffusion model lifecycle** from scratch in PyTorch, targeting production ML engineering practices used at companies like Nvidia, Microsoft, and Google.

**Not a tutorial project** — every component is implemented from scratch:
- DDPM forward/reverse process (no HuggingFace diffusers)
- DPO alignment (no TRL library)
- Manual quantization benchmarking
- Custom inference optimization pipeline

---

## 🏗️ Architecture
CIFAR-10 Dataset (50K images, 32x32)

│

▼

┌─────────────────────┐

│   Stage 1: Train    │  DDPM U-Net (2.1M params)

│   100 epochs        │  Noise schedule, MSELoss, Adam

│   78,200 steps      │  MLflow tracked

└─────────┬───────────┘

│

▼

┌─────────────────────┐

│   Stage 2: Align    │  DPO preference optimization

│   (DPO)             │  Classifier quality judge (77.7% acc)

│                     │  Fixed preference collapse bug

└─────────┬───────────┘

│

▼

┌─────────────────────┐

│  Stage 3: Quantize  │  FP32 → FP16

│                     │  2x size reduction (8MB → 4MB)

│                     │  8% speedup, <5% quality drop

└─────────┬───────────┘

│

▼

┌─────────────────────┐

│   Stage 4: Serve    │  FastAPI REST API

│                     │  /health /generate /metrics

│                     │  p50: 2.56s, p95: 2.84s latency

└─────────────────────┘
---

## 📊 Key Results

### Inference Optimization Benchmark (RTX 5060 Ti)

| Mode | Time | Speedup | Notes |
|------|------|---------|-------|
| PyTorch Eager | 2.43s | 1.00x | Baseline |
| torch.compile (reduce-overhead) | 1.16s | 2.07x | Zero code changes |
| TensorRT FP32 | 0.96s | 2.53x | ONNX → TRT engine |
| TensorRT FP16 | **0.95s** | **2.57x** | Best overall |

### Architecture Benchmark (30 epochs each)

| Architecture | Params | Epoch Time | Loss | Confidence | Visual Quality |
|---|---|---|---|---|---|
| Baseline CNN | 2.10M | 22.5s | 0.050 | 0.496 | ❌ Dark/collapsed |
| Standard Attention | 2.17M | 22.8s | 0.044 | 0.543 | ✅ Good |
| **Linear Attention** | **2.17M** | **23.4s** | **0.011** | **0.552** | **✅ Best** |
| Mamba/SSM | 2.17M | 67.2s | 0.020 | 0.575 | ✅ Rich colors |

### Quantization Results

| Precision | Size | Time | Confidence |
|---|---|---|---|
| FP32 | 8.04 MB | 2.82s | 0.664 |
| FP16 | **4.03 MB** | **2.60s** | 0.632 |

### torch.compile Results

| Mode | Time | Speedup | GPU Memory |
|---|---|---|---|
| Eager | 2.41s | 1.00x | 53.8 MB |
| compile default | 1.61s | 1.50x | 49.1 MB |
| **reduce-overhead** | **1.16s** | **2.07x** | **36.9 MB** |
| max-autotune | 1.16s | 2.07x | 44.9 MB |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch 2.7, CUDA 12.8 |
| Model Serving | FastAPI, Uvicorn |
| Inference Optimization | TensorRT 10.9, ONNX, torch.compile |
| Experiment Tracking | MLflow |
| Containerization | Docker (Nvidia CUDA base image) |
| Orchestration | Kubernetes (2 pods, HPA autoscaler) |
| Pipeline Automation | Apache Airflow (8-task DAG) |
| Monitoring | Custom p50/p95/p99 latency tracking |

---

## 📁 Project Structure
diffusion-lifecycle/

├── src/
│   ├── noise_schedule.py        # Forward process, beta schedule
│   ├── unet.py                  # Baseline CNN U-Net (2.1M params)
│   ├── unet_attention.py        # + Standard self-attention
│   ├── unet_linear_attention.py # + Linear attention (winner)
│   ├── unet_mamba.py            # + Mamba/SSM block
│   ├── dataset.py               # CIFAR-10 loading
│   ├── train.py                 # Training loop (MLflow tracked)
│   ├── sample.py                # DDIM-style reverse process
│   ├── classifier.py            # Quality judge CNN
│   ├── score_samples.py         # Generate + score 100 images
│   ├── dpo.py                   # DPO alignment
│   ├── quantize.py              # FP32 vs FP16 benchmark
│   ├── benchmark.py             # Architecture comparison
│   ├── compile_benchmark.py     # torch.compile optimization
│   ├── tensorrt_benchmark.py    # TensorRT optimization
│   └── serve.py                 # FastAPI + monitoring
├── k8s/
│   ├── deployment.yaml          # 2 replicas, health probes
│   ├── service.yaml             # LoadBalancer port 8000
│   └── hpa.yaml                 # Auto-scale 2-5 pods
├── airflow/
│   └── dags/
│       └── diffusion_retrain.py # 8-task retraining pipeline
├── outputs/
│   ├── model_fp16.pt            # Production model (FP16, 4MB)
│   └── classifier.pt            # Quality judge
├── Dockerfile                   # Nvidia CUDA 12.8 container
├── docker-compose-airflow.yml   # Airflow + PostgreSQL stack
├── requirements.txt
└── README.md

---

## 🚀 Quick Start

### Option 1: Docker (recommended)
```bash
# Build image
docker build -t diffusion-api .

# Run with GPU
docker run --gpus all -p 8000:8000 diffusion-api

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate?num_images=4
curl http://localhost:8000/metrics
```

### Option 2: Local
```bash
conda activate ai-env
python -m src.train          # Train model
python -m src.sample         # Generate images
python -m src.dpo            # DPO alignment
python -m src.quantize       # Quantize to FP16
uvicorn src.serve:app        # Start API
```

### Option 3: Kubernetes
```bash
kubectl apply -f k8s/
kubectl get pods             # Check 2 pods running
# Access at http://localhost:8000
```

### Option 4: Airflow Pipeline (automated)
```bash
docker-compose -f docker-compose-airflow.yml up
# Open http://localhost:8080
# Login: admin/admin
# Trigger: diffusion_lifecycle_retrain DAG
```

---

## 🔬 Key Engineering Insights

**1. DPO Preference Collapse (Debugging Story)**
Initial DPO with `LR=1e-4, EPOCHS=30, BETA=0.1` caused model collapse — mean pixel value shifted from ~0 to ~1.0, producing solid-color outputs. Diagnosed via systematic checkpoint comparison (FP32 vs FP16, before/after DPO). Fixed via stronger KL-anchoring (`BETA=1.0`) and gentler updates (`LR=1e-5`).

**2. Mamba Sequential Scan Bottleneck**
Mamba/SSM achieved best visual quality but ran 3x slower than attention variants due to Python sequential scan loop. Real Mamba requires hardware-optimized CUDA parallel scan kernels — directly demonstrating why CUDA expertise matters for production SSM deployment.

**3. torch.compile Memory Reduction**
`reduce-overhead` mode reduced GPU memory by 31% (53.8MB → 36.9MB) via operator fusion — multiple PyTorch operations compiled into single CUDA kernel calls, eliminating intermediate tensor allocations.

---

## 📈 Monitoring

The API exposes real-time metrics at `/metrics`:

```json
{
  "total_requests": 10,
  "latency_ms": {
    "p50": 2564,
    "p95": 2837,
    "p99": 2837
  },
  "confidence": {
    "mean": 0.648,
    "min": 0.485,
    "max": 0.853
  },
  "gpu_memory_mb": {
    "mean": 19.65,
    "max": 21.96
  }
}
```

---

## 🎯 Nvidia Relevance

| Nvidia Product/Team | This Project |
|---|---|
| TensorRT | TRT export pipeline (2.57x speedup) |
| Triton Inference Server | ONNX export ready for Triton deployment |
| Nvidia Metropolis | Diffusion for synthetic data generation |
| MambaVision (NVlabs) | Mamba/SSM bottleneck implementation |
| Nvidia Earth-2 | Diffusion model architecture knowledge |
| DRIVE Platform | Sequential model understanding (RNN/LSTM/SSM) |

---

## 👤 Author

**Siddhesh Patil**
MS Artificial Intelligence, DePaul University