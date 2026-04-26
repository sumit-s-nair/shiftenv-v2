"""
C2Rust MCPEnvironment — OpenEnv server-side implementation.

Exposes two MCP tools:
  - translate_c_file(c_source, module_name) → JSON with reward details
  - score_rust_code(rust_code, module_name) → reward float

The Rust compiler (cargo check) is the reward oracle. No GPU required.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

# Support running from repo root or as standalone package
_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from reward import compute_reward


def _discover_c_files(data_dir: Optional[str] = None) -> list[Path]:
    """Walk the tests/ directory and return all .c files."""
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    candidates.extend([
        _HERE / "tests",
        Path("tests"),
        Path("."),
    ])
    for root in candidates:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        files = [f for f in root.rglob("*.c") if ".git" not in f.parts]
        if files:
            return sorted(files)
    return []


class C2RustEnvironment(MCPEnvironment):
    """
    OpenEnv environment for C-to-safe-Rust translation.

    Exposes two MCP tools that an RL agent (or a Colab training loop) can call:

      - ``translate_c_file`` — the main training tool. Accepts a Rust
        translation of a C file and returns the compiler reward + diagnostics.

      - ``score_rust_code`` — lightweight scorer. Takes any Rust snippet and
        returns a reward in [0, 1]. Useful for evaluation without episodes.

    The environment also supports ``reset()`` which picks a random .c file
    from the test suite and returns it as the observation for the new episode.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, data_dir: Optional[str] = None):
        mcp = FastMCP("c2rust_env")

        @mcp.tool
        def translate_c_file(rust_code: str, module_name: str = "module") -> str:
            """
            Evaluate a Rust translation of a C module using the compiler.

            Args:
                rust_code: The generated Rust source code to evaluate.
                module_name: The name of the C module being translated.

            Returns:
                JSON string with reward (float), compile_success (bool),
                unsafe_count (int), warning_count (int), error_count (int),
                and compiler_errors (list[str]).
            """
            reward, info = compute_reward(rust_code, module_name=module_name)
            return json.dumps({
                "reward": round(reward, 4),
                "compile_success": info.error_count == 0 and info.compilation_score >= 1.0,
                "unsafe_count": info.unsafe_count,
                "warning_count": info.warning_count,
                "error_count": info.error_count,
                "compiler_errors": info.errors[:10],  # cap at 10 for readability
            })

        @mcp.tool
        def score_rust_code(rust_code: str, module_name: str = "module") -> float:
            """
            Quickly score Rust code quality without starting an episode.

            Args:
                rust_code: Any Rust source code snippet to evaluate.
                module_name: Optional module name hint.

            Returns:
                Reward scalar in [0.0, 1.0]. Higher is better.
            """
            reward, _ = compute_reward(rust_code, module_name=module_name)
            return round(reward, 4)

        super().__init__(mcp)

        self._c_files = _discover_c_files(data_dir)
        self._rng = random.Random()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_file: Optional[Path] = None
        self._files_served = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Pick a random C file and return it as the episode observation."""
        if seed is not None:
            self._rng.seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        if not self._c_files:
            return Observation(
                done=False,
                reward=0.0,
                metadata={"error": "No .c files found. Check that tests/ directory exists."},
            )

        self._current_file = self._rng.choice(self._c_files)
        c_source = self._current_file.read_text(encoding="utf-8", errors="ignore")
        module_name = self._current_file.stem
        self._files_served += 1

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "c_source": c_source,
                "module_name": module_name,
                "file": str(self._current_file),
                "files_available": len(self._c_files),
                "message": (
                    f"Translate '{module_name}.c' to safe Rust. "
                    "Call translate_c_file(rust_code, module_name) to score your attempt."
                ),
            },
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        """Fallback for non-MCP actions."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use call_tool('translate_c_file', ...) or call_tool('score_rust_code', ...)."
                )
            },
        )

    @property
    def state(self) -> State:
        return self._state
