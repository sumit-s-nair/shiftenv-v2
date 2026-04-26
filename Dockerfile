FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. System tools: libclang (for analyzer.py), build tools, Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
        gcc make build-essential libssl-dev libclang-dev git \
        python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python \
    && python --version

# 2. Rust compiler (stable) — installed to /usr/local for cross-user access
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

WORKDIR /app

# 3. Python dependencies for the training loop (includes torch, transformers, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy codebase
COPY . .

# 5. Create non-root user for HF Spaces and set permissions
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

EXPOSE 7860

# Boot: run the actual training loop (engine local + wandb)
CMD ["python3", "main.py", "--engine", "local", "--wandb"]