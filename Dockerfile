# Multi-stage build for authenticated model downloads
FROM python:3.10-slim AS model-downloader

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface_hub

# Set working directory
WORKDIR /model-downloader

# Create directory for downloaded models
RUN mkdir -p /model-downloader/models

# Use ARG to securely pass Hugging Face token at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Login and download the model (secure approach)
RUN if [ -n "$HF_TOKEN" ]; then \
    huggingface-cli login --token ${HF_TOKEN}; \
    huggingface-cli download sesame/csm-1b ckpt.pt --local-dir /model-downloader/models; \
    else echo "No HF_TOKEN provided, model download will be skipped"; fi

# Now for the main application stage
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt /app/

# Ensure pip installs dependencies from requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt  # Fix: Ensure this installs properly

# Copy the rest of the application code
COPY . /app

# Copy downloaded model from the model-downloader stage
COPY --from=model-downloader /model-downloader/models /app/models

# Expose API port
EXPOSE 8000

# Run FastAPI service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
