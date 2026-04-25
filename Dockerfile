FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System build tools + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        gcc make build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python

# Rust toolchain (stable)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && . /root/.cargo/env \
    && rustup component add clippy
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# PyTorch with CUDA 12.4 — separate layer so it is cached independently
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0"

# Training dependencies
RUN pip install --no-cache-dir \
    "transformers==5.6.2" \
    "accelerate==1.13.0" \
    "trl>=0.15.0" \
    "peft==0.18.1" \
    "bitsandbytes==0.49.2" \
    "datasets==4.8.4" \
    "unsloth==2026.4.8" \
    "matplotlib>=3.8.0" \
    "pyyaml>=6.0" \
    "tree-sitter>=0.22.0" \
    "tree-sitter-c>=0.21.0" \
    "wandb>=0.17.0"

# Application code
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "train.py", "--config", "configs/config.yaml"]
