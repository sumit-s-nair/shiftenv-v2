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
# modeling_deepseek.py still imports — we patch it back in code_writer.py.
RUN pip install --no-cache-dir \
    "transformers>=4.46.0" \
    "torch==2.3.1" \
    "bitsandbytes>=0.43.0" \
    "accelerate>=0.30.0" \
    "peft>=0.11.0" \
    "trl>=0.14.0" \
    "datasets>=2.20.0" \
    "gradio>=4.36.0" \
    "pyyaml>=6.0" \
    "tree-sitter>=0.22.0" \
    "tree-sitter-c>=0.21.0" \
    "wandb>=0.17.0"

# Application code
COPY . .

# HuggingFace Spaces expects port 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
EXPOSE 7860

CMD ["python", "-u", "app.py"]
