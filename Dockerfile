# Use Nvidia CUDA base image — matches your CUDA 12.8
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Set working directory
WORKDIR /app

# Install Python 3.12 (already available in Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first (Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY outputs/model_fp16.pt ./outputs/model_fp16.pt
COPY outputs/classifier.pt ./outputs/classifier.pt

# Create outputs directory for metrics
RUN mkdir -p outputs

# Expose FastAPI port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]