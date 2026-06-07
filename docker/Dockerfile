# docker/Dockerfile
# Reproducible training environment for verifiable-alignment
# Base: PyTorch 2.2 + CUDA 12.1

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install flash-attention (requires CUDA headers)
RUN pip install flash-attn --no-build-isolation

# Copy project
COPY pyproject.toml .
COPY . .

# Install project + deps
RUN pip install -e ".[dev]"

# HuggingFace cache dir
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_DIR=/workspace/wandb

# Default: run tests
CMD ["pytest", "tests/", "-v"]
