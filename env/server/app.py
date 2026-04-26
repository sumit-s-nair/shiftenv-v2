"""
FastAPI application for the C2Rust RL Environment.

Exposes the C2RustEnvironment over HTTP and WebSocket using OpenEnv's
create_app helper. Runs on port 7860 for Hugging Face Spaces compatibility.

Usage:
    uvicorn env.server.app:app --host 0.0.0.0 --port 7860
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .c2rust_environment import C2RustEnvironment

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))

app = create_app(
    C2RustEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="c2rust_env",
    max_concurrent_envs=max_concurrent,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
