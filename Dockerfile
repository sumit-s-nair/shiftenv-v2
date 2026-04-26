FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. System tools + libclang-dev (required for analyzer.py) + Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
        gcc make build-essential libssl-dev libclang-dev git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && ln -sf python3.13 /usr/bin/python3 \
    && ln -sf python3.13 /usr/bin/python \
    && python --version

# 2. Rust compiler (stable, needed for reward.py rustc invocations)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && . /root/.cargo/env \
    && rustup component add clippy
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# 3. PyTorch first (CUDA 12.1 wheel — pin exact version matching WSL env)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        "torch==2.5.1" \
        "torchvision" \
        "torchaudio" \
        --index-url https://download.pytorch.org/whl/cu121

# 4. Remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy codebase (after deps so layer is cached on code-only changes)
COPY . .

# HF Spaces: persist LoRA checkpoint dir across restarts via env override
ENV ADAPTER_DIR="/data/lora_adapters"

EXPOSE 7860

# Stream Python output live to HF Spaces logs
ENV PYTHONUNBUFFERED=1

# Boot: health-check server in background, then start training
CMD uvicorn health_check:app --host 0.0.0.0 --port 7860 & \
    python -u main.py --engine local --wandb