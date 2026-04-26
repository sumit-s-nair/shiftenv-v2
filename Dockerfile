FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies, Clang (for AST parsing), and Rust (for rewards)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential libclang-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add clippy

WORKDIR /app

# 2. Install lightweight environment dependencies
# Make sure fastapi, uvicorn, pydantic, and libclang are in this file!
COPY requirements-env.txt .
RUN pip install --no-cache-dir -r requirements-env.txt

# 3. Copy the codebase
COPY . .

# 4. Expose the Hugging Face health check port
EXPOSE 7860

# 5. Boot the FastAPI Environment Server
CMD ["uvicorn", "env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]