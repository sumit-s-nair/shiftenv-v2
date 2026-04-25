FROM python:3.11-slim

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc make build-essential libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (stable)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && . /root/.cargo/env \
    && rustup component add clippy
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Python deps.
# transformers>=4.46 removed is_torch_fx_available which DeepSeek's bundled
# modeling_deepseek.py still imports — train.py patches it back at runtime.
RUN pip install --no-cache-dir \
    "torch==2.4.0" \
    "transformers==4.46.3" \
    "trl==0.15.2" \
    "peft==0.11.0" \
    "accelerate==0.34.2" \
    "bitsandbytes==0.43.0" \
    "datasets==2.21.0" \
    "gradio==4.36.0" \
    "pyyaml>=6.0" \
    "tree-sitter>=0.22.0" \
    "tree-sitter-c>=0.21.0" \
    "wandb>=0.17.0"

# Application code
COPY . .

# Unbuffered output so logs appear immediately in the Space log panel
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "train.py", "--config", "configs/config.yaml"]
