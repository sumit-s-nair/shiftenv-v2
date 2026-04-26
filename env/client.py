"""
C2Rust RL Environment Client.

Connects to a running C2RustEnvironment server (HF Space or local Docker)
and exposes the two MCP tools for use in RL training loops.

Example (connecting to the live HF Space):

    from env.client import C2RustEnv

    with C2RustEnv(base_url="https://YOUR-SPACE.hf.space") as env:
        obs = env.reset()
        c_source = obs.observation.metadata["c_source"]
        module_name = obs.observation.metadata["module_name"]

        # Score your generated Rust code
        result = env.call_tool("translate_c_file",
                               rust_code=my_rust_code,
                               module_name=module_name)
        import json
        info = json.loads(result)
        print(f"Reward: {info['reward']}")
        print(f"Compiled: {info['compile_success']}")

Example (connect from HF Space slug):

    env = C2RustEnv.from_env("YOUR-USERNAME/c2rust-rl")
    ...
"""

from openenv.core.mcp_client import MCPToolClient


class C2RustEnv(MCPToolClient):
    """
    Client for the C2Rust RL environment.

    Inherits full tool-calling interface from MCPToolClient:
      - ``list_tools()``  — discover available tools
      - ``call_tool(name, **kwargs)`` — call a tool by name
      - ``reset()``   — start a new episode (get a fresh C file to translate)

    The two environment tools are:
      - ``translate_c_file(rust_code, module_name)``
          Evaluate a Rust translation. Returns JSON with reward + compiler info.
      - ``score_rust_code(rust_code, module_name)``
          Quickly score any Rust snippet. Returns a float in [0.0, 1.0].
    """
    pass  # MCPToolClient provides all required functionality
