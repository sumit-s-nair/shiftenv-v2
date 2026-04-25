"""
OpenRL-compatible custom environment for C→Rust migration.

The environment presents one C source file at a time.  The agent produces a
Rust translation; the rustc compiler grades it and returns a shaped reward.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional

from tester.compiler import CompilerResult, compile_and_evaluate
from analyzer.static import parse_c_ast
from memory.store import MigrationStore


# ── Observation / action type aliases ───────────────────────────────────────

Observation = dict[str, Any]
Action = dict[str, str]   # {"rust_code": str}
Info = dict[str, Any]


class C2RustEnv:
    """
    Gym-style environment wrapping the C→Rust migration task.

    Compatible with OpenRL's BaseEnv interface (reset / step / close).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str = "data/c_programs",
        max_retries: int = 5,
        timeout: int = 30,
        run_clippy: bool = True,
        store_path: str = "migration_state.json",
        seed: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.run_clippy = run_clippy
        self.store = MigrationStore(store_path)
        self._rng = random.Random(seed)

        self._c_files: list[Path] = sorted(self.data_dir.glob("*.c"))
        if not self._c_files:
            raise FileNotFoundError(f"No .c files found in {self.data_dir}")

        # Per-episode state
        self._current_file: Optional[Path] = None
        self._c_source: str = ""
        self._c_ast: dict = {}
        self._reference_output: str = ""
        self._previous_rust: str = ""
        self._compiler_errors: list = []
        self._error_count: int = 0
        self._retry_count: int = 0
        self._done: bool = False

    # ── Gym API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[Observation, Info]:
        if seed is not None:
            self._rng.seed(seed)

        # Pick a C file that hasn't been successfully migrated yet
        pending = [
            f for f in self._c_files
            if not self.store.is_migrated(f.stem)
        ]
        if not pending:
            pending = self._c_files  # all done — cycle again

        self._current_file = self._rng.choice(pending)
        self._c_source = self._current_file.read_text(encoding="utf-8")
        self._c_ast = parse_c_ast(self._c_source)

        ref_path = self._current_file.with_suffix(".ref.txt")
        self._reference_output = ref_path.read_text(encoding="utf-8") if ref_path.exists() else ""

        self._previous_rust = ""
        self._compiler_errors = []
        self._error_count = 0
        self._retry_count = 0
        self._done = False

        obs = self._build_obs()
        info: Info = {"file": str(self._current_file), "action": "reset"}
        return obs, info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Info]:
        """
        action: {"rust_code": str}

        Returns (observation, reward, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode — call reset() first")

        rust_code: str = action.get("rust_code", "")
        module = self._current_file.stem if self._current_file else "unknown"

        result: CompilerResult = compile_and_evaluate(
            rust_code=rust_code,
            reference_output=self._reference_output or None,
            timeout=self.timeout,
            run_clippy=self.run_clippy,
        )

        # Update episode state
        self._previous_rust = rust_code
        self._compiler_errors = [
            {
                "message": e.message,
                "code": e.code,
                "level": e.level,
                "line": e.line,
                "column": e.column,
                "rendered": e.rendered,
            }
            for e in result.errors
        ]
        self._error_count = len(result.errors)
        self._retry_count += 1

        # Persist to store
        self.store.update(
            module=module,
            success=result.success,
            retry_count=self._retry_count,
            error_type=result.error_type,
            reward=result.reward,
            rust_code=rust_code if result.success else None,
        )

        # Termination conditions
        terminated = result.success and result.test_passed is not False
        truncated = self._retry_count >= self.max_retries

        if terminated or truncated:
            self._done = True

        obs = self._build_obs()
        info: Info = {
            "error_type": result.error_type,
            "error_line": result.error_line,
            "compiler_output": result.raw_stderr,
            "unsafe_count": result.unsafe_count,
            "test_passed": result.test_passed,
            "clippy_clean": result.clippy_clean,
            "module": module,
        }

        return obs, result.reward, terminated, truncated, info

    def close(self) -> None:
        self.store.save()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_obs(self) -> Observation:
        return {
            "c_source": self._c_source,
            "c_ast": self._c_ast,
            "previous_rust": self._previous_rust,
            "compiler_errors": self._compiler_errors,
            "error_count": self._error_count,
            "retry_count": self._retry_count,
            "migration_context": self.store.get_context(),
        }

    # ── OpenRL compatibility shim ────────────────────────────────────────────

    @property
    def observation_space(self) -> dict:
        return {"type": "dict", "description": "C→Rust migration observation"}

    @property
    def action_space(self) -> dict:
        return {"type": "dict", "description": "Generated Rust code"}

    @property
    def reward_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"C2RustEnv(data_dir={self.data_dir!r}, "
            f"files={len(self._c_files)}, "
            f"max_retries={self.max_retries})"
        )
