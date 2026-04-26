FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Tools AND libclang-dev (Crucial for analyzer.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
        gcc make build-essential libssl-dev libclang-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python

# 2. Install the Rust Compiler
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && . /root/.cargo/env \
    && rustup component add clippy
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# 3. Install Python Dependencies (Force CUDA PyTorch first)
COPY requirements.txt .
RUN pip install --no-cache-dir "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your codebase
COPY . .

# Force Python logs to stream live to the Hugging Face UI
ENV PYTHONUNBUFFERED=1

# 5. Boot Training (with WandB logging enabled)
CMD ["python", "-u", "main.py", "--engine", "local", "--wandb"]