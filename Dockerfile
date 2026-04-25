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

# Python deps (copy first for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    openai \
    gradio \
    pyyaml \
    tree-sitter \
    tree-sitter-c \
    datasets \
    wandb

# Application code
COPY . .

# HuggingFace Spaces expects port 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
EXPOSE 7860

CMD ["python", "app.py"]
